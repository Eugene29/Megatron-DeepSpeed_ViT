# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.

"""Pretrain VIT"""
from mpi4py import MPI
comm = MPI.COMM_WORLD
comm.Barrier()
import deepspeed.comm as dist

import torch
import torch.nn.functional as F
from functools import partial
from megatron import get_args, get_timers, print_rank_0
from megatron.core.enums import ModelType
from megatron.core import parallel_state as mpu, tensor_parallel
from megatron.data.vit_dataset import build_train_valid_datasets
from megatron.model.vision.classification import VitClassificationModel
from megatron.model.vision.classification import MitClassificationModel
from megatron.core.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy
from megatron.training import pretrain
from megatron.utils import average_losses_across_data_parallel_group
from megatron.arguments import core_transformer_config_from_args
import deepspeed
import os
# from deepspeed.runtime.utils import see_memory_usage


def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    args = get_args()
    config = core_transformer_config_from_args(args)
    # see_memory_usage(f"Before Building Model", force=True)

    ##TODO: enable PP here
    if hasattr(mpu, 'get_sequence_data_parallel_group'):
        dpg = mpu.get_sequence_data_parallel_group()
    elif hasattr(mpu, 'get_data_parallel_group'):
        dpg = mpu.get_data_parallel_group()
    else:
        dpg = None

    if args.use_MICS:
        zero_init = deepspeed.zero.MiCS_Init
    else:
        zero_init = deepspeed.zero.Init
    with zero_init(data_parallel_group=dpg,
                             remote_device=None if args.remote_device == 'none' else args.remote_device,
                             config_dict_or_path=args.deepspeed_config_dict,
                             enabled=args.zero_stage == 3,
                             mpu=mpu):
        if args.vision_backbone_type == 'vit':
            print_rank_0("building VIT model ...")
            model = VitClassificationModel(config=config,
                                        num_classes=args.num_classes,
                                        pre_process=pre_process,
                                        post_process=post_process)
        elif args.vision_backbone_type == 'mit':
            print_rank_0("building MIT model ...")
            model = MitClassificationModel(num_classes=args.num_classes,
                                        pre_process=pre_process,
                                        post_process=post_process)
        else:
            raise Exception('{} vision backbone is not supported.'.format(
                                args.vision_backbone_type))
    # see_memory_usage(f"After Building Model", force=True)
    return model


def get_batch(data_iterator):
    """Generate a batch.

    Args:
        data_iterator: Iterable dataset.

    Returns:
        sample: A data sample with images, tokens, etc.
    """
    args = get_args()
    # rank = args.rank
    dp = mpu.get_data_parallel_world_size()
    # dp_group = mpu.get_data_parallel_group()
    dp_rank = mpu.get_data_parallel_rank()
    dp_src_rank = mpu.get_data_parallel_src_rank()

    assert args.fp16 or args.bf16
    img_dtype = torch.float16 if args.fp16 else torch.bfloat16

    ## Generate Random TOY dataset
    if os.environ["DATA"] == "TOY":
        ## 1. First, only rank0 generates the data
        ## 2. rank0 scatters data to other sp_rank=1

        dev = deepspeed.accelerator.get_accelerator().current_device()

        b = int(os.environ["GBS"])
        c = args.num_channels
        h = int(os.environ["IMG_W"])
        w = int(os.environ["IMG_H"])

        MBS = int(os.environ["MBS"])

        assert MBS == b / dp, f"Environment Var MBS (GBS ({b})/ DP ({dp}))is not local MBS: ({MBS})"
        assert b % dp == 0, "global batch size is not divisible by dp degree"

        # img_dtype = torch.float16
        label_dtype = torch.int64

        if dp_src_rank == 0: ## only need data in first dp group as it will get broadcasted to other dp group. 
            ## Generate TOY DATASET on rank0
            if "VIT3D" not in os.environ:
                full_img = torch.randn(b, c, h, w, dtype=img_dtype, device=dev) ## B, S
            else:
                d = int(os.environ["IMG_D"])
                full_img = torch.randn(b, c, h, w, d, dtype=img_dtype, device=dev)
            num_classes = int(os.environ["NUM_CLASSES"])
            full_label = torch.randint(num_classes, (b,), dtype=label_dtype, device=dev) ## B, S

            ## Partition data to replicate DP mechanism. 
            strt_idx = MBS * dp_rank
            end_idx = strt_idx + MBS
            data_dict = {'image': full_img[strt_idx: end_idx], 'label': full_label[strt_idx: end_idx]}
        else:
            data_dict = None
    else:
        # Broadcast data.
        if data_iterator is not None:
            data = next(data_iterator)
            data_dict = {}
            data_dict['image'] = data[0]
            data_dict['label'] = data[1]
        else:
            data_dict = None

    ## Log data (only from the first dp group)
    if "DATA_PATH_LOG" in os.environ and dp_src_rank == 0:
        with open(os.environ["DATA_PATH_LOG"], mode='a') as file:
            file.write(f"img: {data_dict['image']}\n")
            file.write(f"label: {data_dict['label']}\n")

    data_i = tensor_parallel.broadcast_data(["label"], data_dict, torch.int64)
    data_f = tensor_parallel.broadcast_data(["image"], data_dict, img_dtype)

    labels = data_i["label"].long().contiguous()
    images = data_f["image"].contiguous()

    return images, labels


def loss_func(labels, output_tensor):
    sp_rank = mpu.get_sequence_parallel_rank()
    logits = output_tensor.contiguous()
    outputs = torch.argmax(logits, -1)
    correct = (outputs == labels).float()
    accuracy = torch.mean(correct)
    loss = F.cross_entropy(logits, labels)
    if sp_rank != 0:
        ## DROPOUT ALL, cut off gradients
        loss = loss * 0 
    averaged_loss = average_losses_across_data_parallel_group([loss, accuracy])
    return loss, {"loss": averaged_loss[0], "accuracy": averaged_loss[1]}


def forward_step(data_iterator, model):
    """Forward step."""
    timers = get_timers()

    # Get the batch.
    timers("batch-generator", log_level=2).start()
    (
        images,
        labels,
    ) = get_batch(data_iterator)
    timers("batch-generator").stop()

    # Forward model. lm_labels
    output_tensor = model(images)

    return output_tensor, partial(loss_func, labels)

def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0(
        "> building train, validation, and test datasets " "for VIT ..."
    )
    train_ds, valid_ds = build_train_valid_datasets(
        data_path=args.data_path,
        image_size=(args.img_h, args.img_w)
    )
    print_rank_0("> finished creating VIT datasets ...")

    return train_ds, valid_ds, None


if __name__ == "__main__":
    import ezpz as ez
    RANK = ez.setup_torch(backend="deepspeed")
    WORLD_SIZE = ez.get_world_size()
    LOCAL_RANK = ez.get_local_rank()
    DEVICE_TYPE = ez.dist.get_torch_device_type()

    if torch.cuda.is_available():
        torch.cuda.set_device(LOCAL_RANK)

    import time
    from megatron import get_wandb_writer
    train_strt = time.time()
    # try:
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'dataloader_type': 'cyclic', 'vision_pretraining': True}
    )

    dist.barrier() ## prevent accidental saving? 
    print_rank_0(f"tot train time: {time.time() - train_strt}")

    if dist.get_rank() == 0:
        import pprint
        args = get_args()
        log_keys = [ "iteration", "time", "LLM_TFLOPS", "TFLOPS", "TFLOPS_per_gpu", "samples_per_sec", "memory_fpt(GiB)" ]
        log_dict = {k:getattr(args, k) for k in log_keys}
        log_dict["num_params"] = os.environ.get("NUM_PARAMS", "NA")
        log_dict["model_size"] = os.environ.get("VIT", "NA")
        wandb_writer = get_wandb_writer()
        wandb_writer.log(log_dict, step=args.logger_iteration)

        print("Pretrain completed.")
    print_rank_0("switching stderr to /dev/null to prevent endless stream of 'longjmp'")
    import sys
    sys.stderr = open(os.devnull, "w")

    exit()