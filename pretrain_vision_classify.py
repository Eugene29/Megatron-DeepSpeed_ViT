# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.

"""Pretrain VIT"""
from mpi4py import MPI
import torch.distributed as dist
import torch.distributed
comm = MPI.COMM_WORLD
comm.Barrier()

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
# from deepspeed.runtime.utils import see_memory_usage

def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    args = get_args()
    config = core_transformer_config_from_args(args)
    # see_memory_usage(f"Before Building Model", force=True)

    ## Something related to zero if you are using seq parallel. Feeding it to deepspeed zero?
    # if hasattr(mpu, 'get_sequence_data_parallel_group'):
    #     dpg = mpu.get_sequence_data_parallel_group()
    # elif hasattr(mpu, 'get_data_parallel_group'):
    #     dpg = mpu.get_data_parallel_group()
    # else:
    #     dpg = None
    
    ##TODO: Check what this does for ZERO
    # with deepspeed.zero.Init(data_parallel_group=dpg,
    #                         remote_device=None if args.remote_device == 'none' else args.remote_device,
    #                         config_dict_or_path=args.deepspeed_config_dict,
    #                         enabled=args.zero_stage == 3,
    #                         mpu=mpu):
        ##TODO: enable PP here. 
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
    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
        data_dict = {}
        data_dict['image'] = data[0]
        data_dict['label'] = data[1]
    else:
        data_dict = None

    data_i = tensor_parallel.broadcast_data(["label"], data_dict, torch.int64) ##TODO: lower precision, will it get angry at me if I set it to 16 or 32? 
    data_f = tensor_parallel.broadcast_data(["image"], data_dict, torch.float16) ## images are in int8 -> fp16

    labels = data_i["label"].long().contiguous()
    images = data_f["image"].contiguous()

    return images, labels

def loss_func(labels, output_tensor):
    # def gather_last_dim_from_sequence_parallel_region(input_, tensor_parallel_output_grad=True):
    #     return _GatherFromSequenceParallelRegion.apply(input_, tensor_parallel_output_grad)

    seq_rank = mpu.get_sequence_parallel_rank()
    # # print(f"seq_rank: {mpu.get_sequence_parallel_rank()}")
    # logits = output_tensor.contiguous().float()
    # # print(f"logits: {logits}")
    # loss = F.cross_entropy(logits, labels)
    # args = get_args()
    # labels = F.one_hot(labels, num_classes=args.num_classes)
    # labels = labels.unsqueeze(0)
    # output_tensor = output_tensor.unsqueeze(0)
    # output_tensor = output_tensor.T.unsqueeze(-1) ## TODO: is .T as efficient as .transpose?

    # if mpu.get_sequence_parallel_rank() == 0:
    # print(f"labels.shape: {labels.shape}")
    # print(f"output_tensor.shape: {output_tensor.shape}")
    logits = output_tensor.contiguous().float()
    outputs = torch.argmax(logits, -1)
    correct = (outputs == labels).float()
    accuracy = torch.mean(correct)
    # if seq_rank > 1:
    #     dist.broadcast(logits, src=0, group=mpu.get_sequence_parallel_group())
    # from megatron.core.sequence_parallel import vocab_sequence_parallel_cross_entropy
    # from megatron.core.tensor_parallel import vocab_parallel_cross_entropy
    loss = vocab_parallel_cross_entropy(logits.contiguous(), labels, for_vit=True).mean()
    import os
    debug_fname = os.environ["DEBUG_FNAME"]
    if debug_fname != "None":
        with open(debug_fname, "a") as f:
            f.write(f"[{seq_rank}] output after head: {output_tensor}\n")
            f.write(f"[{seq_rank}] real losses_reduced: {loss}\n")

    # if seq_rank==0:
    #     logits = logits.T
    #     # torch.distributed.gather(logits, dst=0, group=group)
    #     # from megatron.core.tensor_parallel import mappings 
    #     # from megatron.core.tensor_parallel.mappings import gather_from_tensor_model_parallel_region, dummy_function
    #     from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_group
    #     from megatron.core import tensor_parallel
    #     # import gather_from_sequence_parallel_group
    #     logits = tensor_parallel.gather_from_sequence_parallel_group(logits).T.contiguous() ## Gather first dim
    #     logits = tensor_parallel.gather_from_tensor_model_parallel_region(logits).T.contiguous() ## Gather first dim
        
    #     print(f"logits.shape: {logits.shape}")
    #     print(f"labels.shape: {labels.shape}")
    #     loss = F.cross_entropy(logits, labels)
    #     # outputs = torch.argmax(logits, -1)
    #     # correct = (outputs == labels).float()
    #     # accuracy = torch.mean(correct)
    # else:
    #     loss = torch.empty(1, device=torch.cuda.current_device())
    #     accuracy = torch.empty(1, device=torch.cuda.current_device())
    # group = mpu.get_sequence_parallel_group()
    # dist.barrier(group=group)
    # # dist.broadcast(loss, src=0, group=group)
    # # dist.broadcast(accuracy, src=0, group=group)

    # with open(debug_fname, "a") as f:
    #     f.write(f"[{seq_rank}] Got hung here?\n")

    # with open(debug_fname, "a") as f:
    #     f.write(f"[{seq_rank}] output after head: {output_tensor}\n")

    # with open(debug_fname, "a") as f:
    #     f.write(f"[{seq_rank}] real losses_reduced: {loss}\n")
    # raise KeyboardInterrupt("break")

    # print(f"loss: {loss}")
    # print(f"loss.shape: {loss.shape}")
    # print(f"loss.grad_fn: {loss.grad_fn}")
    # print(f"loss.device: {loss.device}")
    # print(f"outputs.device: {outputs.device}")
    # print(f"accuracy.device: {accuracy.device}")
    # else:
        # print("yahoo")
        # loss = torch.empty(1, device=torch.cuda.current_device())
        # accuracy = torch.empty(1, device=torch.cuda.current_device())
        # loss = torch.zeros(1, device=torch.cuda.current_device())
        # accuracy = torch.zeros(1, device=torch.cuda.current_device())
    # dist.broadcast(loss, src=0, group=mpu.get_sequence_parallel_group())
    # dist.broadcast(accuracy, src=0, group=mpu.get_sequence_parallel_group())
    # dist.all_reduce(loss, op=dist.ReduceOp.SUM, group=mpu.get_sequence_parallel_group())
    # dist.all_reduce(accuracy, op=dist.ReduceOp.SUM, group=mpu.get_sequence_parallel_group())


    # print(f"loss: {loss}")
    # print(f"loss.grad_fn: {loss.grad_fn}")
    # print(f"loss[0].grad_fn: {loss[0].grad_fn}")

    # raise KeyError("breakdance")

    # comm.Barrier()
    # print(f"[{seq_rank}] loss: {loss}")
    # print(f"[{seq_rank}] accuracy: {accuracy}")
    ## output, logits, loss: [b, ]
    # loss = vocab_parallel_cross_entropy(logits.contiguous(), labels).mean() ##TODO: implement later for better throughput
    # print_rank_0(f"output_tensor.shape: {output_tensor.shape}")
    # print_rank_0(f"logits.shape: {logits.shape}")
    # print_rank_0(f"labels.shape: {labels.shape}")
    

    ## TODO: Could only run on 1 of the SP group.
    # with torch.no_grad():
    #     gathered_logits = gather_from_sequence_parallel_group(logits)
    #     outputs = torch.argmax(gathered_logits, -1)
    #     correct = (outputs == labels).float()
    #     accuracy = torch.mean(correct)
    averaged_loss = average_losses_across_data_parallel_group([loss, accuracy])
    # averaged_loss = loss, accuracy

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
    ##TODO: What's going on under the hood? Take time to replace it with MPI?  
    import ezpz as ez
    RANK = ez.setup_torch(backend="deepspeed")#, timeout=72000) ## 20 hours max.
    WORLD_SIZE = ez.get_world_size()
    LOCAL_RANK = ez.get_local_rank()
    DEVICE_TYPE = ez.dist.get_torch_device_type()
    if torch.cuda.is_available():
        torch.cuda.set_device(LOCAL_RANK)

    # RANK = comm.Get_rank()
    # WORLD_SIZE = comm.Get_size()
    # LOCAL_RANK = RANK % WORLD_SIZE
    # # torch.distributed.init_process_group(backend="deepspeed")
    # torch.distributed.init_process_group(backend="deepspeed", init_method="env://", world_size=WORLD_SIZE, rank=RANK)
    # ##Q. when is the above neccessary? pretrain_gpt for example, doesn't have any torch.distributed.init_process_group
    # torch.distributed.barrier()

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'dataloader_type': 'cyclic', 'vision_pretraining': True}
    )
    print("Pretrain completed.")
    exit()
