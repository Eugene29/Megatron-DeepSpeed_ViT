"""Pretrain SWIN VIT"""
from mpi4py import MPI
comm = MPI.COMM_WORLD
comm.Barrier()

from megatron.initialize import initialize_megatron
from megatron import get_args, get_timers, print_rank_0
from megatron.core.enums import ModelType
from megatron.core import parallel_state as mpu, tensor_parallel
from megatron.data.vit_dataset import build_train_valid_datasets
from megatron.core.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy
from megatron.training import pretrain
from megatron.utils import average_losses_across_data_parallel_group
from megatron.arguments import core_transformer_config_from_args
from megatron.model.vision.swin_backbone_alcf import SwinTransformerV2Cr
import deepspeed
import deepspeed.comm as dist

import torch
import torch.nn.functional as F
# import deepspeed
# import deepspeed.comm as dist

import os
from functools import partial

def model_provider(pre_process=True, post_process=True):
    args = get_args()
    config = core_transformer_config_from_args(args)
    assert pre_process and post_process
    ## Merge sequence parallel group with data parallel group for ZERO
    if hasattr(mpu, "get_sequence_data_parallel_group"):
        data_parallel_group = mpu.get_sequence_data_parallel_group()
    elif hasattr(mpu, "get_data_parallel_group"):
        data_parallel_group = mpu.get_sequence_data_parallel_group()
    else:
        data_parallel_group = None
    ## Create Zero context(?)
    if args.use_MICS:
        zero_init = deepspeed.zero.MICS_Init
    else:
        zero_init = deepspeed.zero.Init
    remote_device = None if args.remote_device == 'none' else args.remote_device
    with zero_init(data_parallel_group=data_parallel_group,
                   remote_device=remote_device,
                   config_dict_or_path=args.deepspeed_config_dict,
                   enabled=args.zero_stage == 3, 
                   mpu=mpu):
        ## TODO: connect model to reflect args?
        mlp_ratio = args.ffn_hidden_size / args.hidden_size
        model = SwinTransformerV2Cr(
            config=config,
            img_size=[args.img_h, args.img_w],
            patch_size=args.patch_dim,
            depths=[args.num_layers],
            num_heads=(args.num_attention_heads,),
            in_chans=args.num_channels,
            out_chans=args.num_channels,
            embed_dim=args.hidden_size,
            img_window_ratio=args.swin_window2image_ratio,  # Modify number of windows
            window_size=args.swin_window_size,
            drop_path_rate=0,  # Stochastic Depth
            full_pos_embed=True,  # TODO: Replace with ROPE?
            rel_pos=False,  # TODO: REMOVE from args?
            mlp_ratio=mlp_ratio,  # Fixed projection dimension
            checkpoint_stages=False,  # TODO: Enable activation checkpointing
            residual=False,  # TODO: What is residual doing?
        )
    return model

def get_batch(data_iterator):
    args = get_args()
    dp = mpu.get_data_parallel_world_size()
    # sp = mpu.get_    
    dp_rank = mpu.get_data_parallel_rank()
    dp_src_rank = mpu.get_data_parallel_src_rank()


    assert args.fp16 or args.bf16
    img_dtype = torch.float16 if args.fp16 else torch.bfloat16

    ## Generate Random TOY dataset
    if args.use_toy_data:
        ## 1. First, only rank0 generates the data
        ## 2. rank0 scatters data to other sp_rank=1
        dev = deepspeed.accelerator.get_accelerator().current_device()
        ## TODO: Change these environment variables with args
        b = int(os.environ["GBS"])
        c = args.num_channels
        h = int(os.environ["IMG_W"])
        w = int(os.environ["IMG_H"])
        MBS = int(os.environ["MBS"])
        # label_dtype = torch.int64
        assert MBS == b / dp, f"Environment Var MBS (GBS ({b})/ DP ({dp}))is not local MBS: ({MBS})"
        assert b % dp == 0, "global batch size is not divisible by dp degree"
        data_dict = None
        if dp_src_rank == 0:  # First DP group will broadcast to other dp groups
            ## Generate TOY DATASET on rank0
            # if "VIT3D" not in os.environ:
            # else:
            # d = int(os.environ["IMG_D"])
            # full_img = torch.randn(b, c, h, w, d, dtype=img_dtype, device=dev)
            num_classes = int(os.environ["NUM_CLASSES"])
            full_img = torch.randn(b, c, h, w, dtype=img_dtype, device=dev)
            full_label = torch.randn_like(full_img)
            # full_label = torch.randint(num_classes, (b,), dtype=label_dtype, device=dev) ## B, S
            ## Partition data to replicate DP mechanism. 
            strt_idx = MBS * dp_rank
            end_idx = strt_idx + MBS
            data_dict = {
                'image': full_img[strt_idx: end_idx], 
                'label': full_label[strt_idx: end_idx]
            }
    else:
        # Broadcast data.
        if data_iterator is not None:
            data = next(data_iterator)
            data_dict = {}
            data_dict['image'] = data[0]
            data_dict['label'] = data[0]  # Use input as output for autoencoder
            # data_dict['label'] = data[1]
        else:
            data_dict = None

    data_i = tensor_parallel.broadcast_data(["label"], data_dict, img_dtype) ##TODO: lower precision, will it get angry at me if I set it to 16 or 32? 
    data_f = tensor_parallel.broadcast_data(["image"], data_dict, img_dtype)
    labels = data_i["label"].contiguous()
    images = data_f["image"].contiguous()
    return images, labels

def loss_func(labels, output_tensor):
    sp_rank = mpu.get_sequence_parallel_rank()
    ## TODO: How was bfloat working with .float()?
    logits = output_tensor.contiguous()
    # outputs = torch.argmax(logits, -1)
    # correct = (outputs == labels).float()
    with torch.no_grad():
        mae_loss = F.l1_loss(logits, labels)
    # with torch.no_grad():
    #     mean_loss = torch.mean(correct)
    loss = F.mse_loss(logits, labels)
    if sp_rank != 0:
        ## DROPOUT ALL, cut off gradients
        loss = loss * 0 
    ## TODO: Q. Why doesn't the below ruin our loss and acc as they get reduced across by "noise tokens"? (DP vs. SP looks perfect). Maybe the below isn't whats visualized on wandb? 
    ## FIXME: Bring back the original result
    # averaged_loss = [0, 0]
    averaged_loss = average_losses_across_data_parallel_group([loss, mae_loss])
    return loss, {"loss": averaged_loss[0], "mae_loss": averaged_loss[1]}

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

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'dataloader_type': 'cyclic', 'vision_pretraining': True}
    )

    # initialize_megatron(
    #     extra_args_provider={},
    #     args_defaults={'dataloader_type': 'cyclic', 'vision_pretraining': True}, 
    #     external_args={}
    # )
    # args = get_args()

    # model = get_model(model_provider_func, model_type)
    # model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
    #     model_provider, model_type, teacher=False, data_post_process=data_post_process,
    #     build_train_valid_test_datasets_provider=None
    # )

    # args = get_args()
    # embed_dim = args.hidden_dim
    # depths = [2, 0, 0, 0]
    # num_heads = [args.num_, 0, 0, 0]
    # window_size = 16
    # drop_path_rate = 0 ## stochastic depth
    # output_avg = False  # average output channel?

    # dist.breakpoint()
    # model = get_swin()

    # img_size = [32, 32]  # CIFAR
    # patch_size = 16
    # window_size = 8
    # embed_dim = 128
    # depths = [24]
    # num_heads = [12]
    # model = SwinTransformer(
    #     img_size=img_size,
    #     in_chans=3,
    #     patch_size=patch_size,
    #     embed_dim=embed_dim,
    #     depths=depths,
    #     num_heads=num_heads,
    #     window_size=window_size,
    #     drop_path_rate=0,
    #     output_avg=False,
    # )

    ## Try out swin transformer
    ## get an output
    ## What would it take to utilize MDS parallelism framework? 
        ## Probably just need to partition with sp group and match 
        ## how each layers are called.
    ## Optimize wout parallelism
    ## Optimize with parallelism

    # def get_swin(drop_path_rate=0, output_avg=False):
    # args = get_args()

    # window_size = 7
    # embed_dim = 128
    # depths = [2, 2, 18, 2]
    # num_heads = [4, 8, 16, 32]

    # from megatron.model.vision.swin_backbone_alcf import swinv2net_megatron_deepspeed
    # model = swinv2net_megatron_deepspeed()
    # torch.input
    # print(f"model: {model}", flush=True)