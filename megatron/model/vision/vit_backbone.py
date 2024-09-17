# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Vision Transformer(VIT) model."""

import math
import einops
import torch
import apex
import torch.distributed
import torch.nn.functional as F
from megatron import get_args
from megatron.model.transformer import ParallelTransformer
from megatron.core import parallel_state as mpu
from megatron.model.utils import (
    get_linear_layer,
    init_method_normal,
    scaled_init_method_normal,
)
from megatron.model.module import MegatronModule
from megatron import get_args, print_rank_0

CLASS_TOKEN_LENGTH = 1 # 8

class VitMlpHead(MegatronModule):
    """Pooler layer.

    Pool hidden states of a specific token (for example start of the
    sequence) and add a linear transformation followed by a tanh.

    Arguments:
        hidden_size: hidden size
        init_method: weight initialization method for the linear layer.
            bias is set to zero.
    """

    def __init__(self, hidden_size, num_classes):
        super(VitMlpHead, self).__init__()
        self.dense_in = torch.nn.Linear(hidden_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.dense_out = torch.nn.Linear(hidden_size, num_classes)
        torch.nn.init.constant_(self.dense_out.bias, -10)

    def forward(self, hidden_states):
        # hidden_states: [b, 1, h]
        # sequence_index: index of the token to pool.
        dense_in_result = self.dense_in(hidden_states)
        tanh_result = torch.tanh(dense_in_result)
        dense_out_result = self.dense_out(tanh_result)
        return dense_out_result


def isPerfectSquare(x):
    if(x >= 0):
        sr = math.sqrt(x)
        return (int(sr) * int(sr) == x)
    return False


def twod_interpolate_position_embeddings_hook(
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):

    args = get_args()
    num_patches_per_dim_h = args.img_h // args.patch_dim
    num_patches_per_dim_w = args.img_w // args.patch_dim
    num_patches = num_patches_per_dim_h * num_patches_per_dim_w
    hidden_size = args.hidden_size

    key = prefix + "weight"

    assert key in state_dict
    if key in state_dict:
        input_param = state_dict[key]

        input_seq_len = input_param.shape[0]
        assert(isPerfectSquare(input_seq_len) or isPerfectSquare(input_seq_len - CLASS_TOKEN_LENGTH))
        input_has_class_token = not isPerfectSquare(input_seq_len)
        num_tok_input = input_seq_len - CLASS_TOKEN_LENGTH if input_has_class_token else input_seq_len
        num_tok_output = num_patches
        output_has_class_token = args.class_token_present

        # update input_param and load it to state_dict[key]
        if input_has_class_token:
            input_param_tok = input_param[:CLASS_TOKEN_LENGTH, :]
            input_param_grid = input_param[CLASS_TOKEN_LENGTH:, :]
        else:
            input_param_tok = torch.zeros(CLASS_TOKEN_LENGTH, hidden_size)
            input_param_grid = input_param

        assert input_param.shape[1] == hidden_size

        if num_tok_input != num_tok_output:

            gs_input = int(math.sqrt(num_tok_input))
            gs_new = (num_patches_per_dim_h, num_patches_per_dim_w)

            input_param_grid = input_param_grid.transpose(0, 1).contiguous()
            input_param_grid = input_param_grid.reshape(
                (1, -1, gs_input, gs_input)
            )
            input_param_grid = input_param_grid.float()
            scale_factor = (gs_new[0] / gs_input, gs_new[1] / gs_input)

            input_param_grid = F.interpolate(
                input_param_grid, scale_factor=scale_factor, mode="bilinear"
            )

            input_param_grid = input_param_grid.half()
            input_param_grid = input_param_grid.reshape((-1, num_tok_output))
            input_param_grid = input_param_grid.transpose(0, 1).contiguous()

            assert input_param_grid.shape[1] == hidden_size

        input_param = input_param_grid
        assert (
            input_param.shape[0] == num_tok_output
            and input_param.shape[1] == hidden_size
        )

        if output_has_class_token:
            input_param = torch.cat((input_param_tok, input_param), dim=0)

        state_dict[key] = input_param


class VitBackbone(MegatronModule):
    """Vision Transformer Model."""

    def __init__(self,
                 config,
                 pre_process=True,
                 post_process=True,
                 class_token=True,
                 single_token_output=False,
                 post_layer_norm=True,
                 drop_path_rate=0.0):
        super(VitBackbone, self).__init__(share_embeddings_and_output_weights=False)
        args = get_args()

        self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy

        self.pre_process = pre_process
        self.post_process = post_process
        self.class_token = class_token
        self.post_layer_norm = post_layer_norm
        self.hidden_size = args.hidden_size
        self.patch_dim = args.patch_dim
        self.img_h = args.img_h
        self.img_w = args.img_w
        self.micro_batch_size = args.micro_batch_size
        self.single_token_output = single_token_output
        self.drop_path_rate = drop_path_rate
        ## sequence parallelism
        self.ds_sequence_parallel = args.ds_sequence_parallel_size > 1

        assert self.img_h % self.patch_dim == 0, f"img_h:, {self.img_h}, patch_dim: {self.patch_dim}"
        assert self.img_w % self.patch_dim == 0, f"img_w:, {self.img_w}, patch_dim: {self.patch_dim}"
        self.num_patches_per_dim_h = self.img_h // self.patch_dim
        self.num_patches_per_dim_w = self.img_w // self.patch_dim
        self.num_patches = self.num_patches_per_dim_h * self.num_patches_per_dim_w
        self.seq_length = self.num_patches + (CLASS_TOKEN_LENGTH if self.class_token else 0)
        self.flatten_dim = self.patch_dim * self.patch_dim * args.num_channels
        self.input_tensor = None
        self.position_ids = None

        if self.pre_process:
            # cls_token
            if self.class_token:
                self.cls_token = torch.nn.Parameter(
                    torch.randn(1, CLASS_TOKEN_LENGTH, self.hidden_size)
                )
                torch.nn.init.zeros_(self.cls_token)
            self.position_ids = torch.arange(self.seq_length).expand(1, -1).cuda()
            
            # Linear encoder
            self.linear_encoder = torch.nn.Linear(
                self.flatten_dim, self.hidden_size
            )

            # embedding
            self.position_embeddings = torch.nn.Embedding(
                self.seq_length, self.hidden_size
            )
            init_method_normal(args.init_method_std)(
                self.position_embeddings.weight
            )

            args.class_token_present = self.class_token
            self.position_embeddings._register_load_state_dict_pre_hook(
                twod_interpolate_position_embeddings_hook
            )

            self.embedding_dropout = torch.nn.Dropout(args.hidden_dropout)

        # Transformer
        self.transformer = ParallelTransformer(
            config,
            model_type=args.model_type, ## Added agnostic model_type argument
            pre_process=self.pre_process,
            post_process=self.post_process,
            post_layer_norm=self.post_layer_norm,
            drop_path_rate=self.drop_path_rate
        )

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        self.transformer.set_input_tensor(input_tensor)

    def forward(self, input):
        if self.pre_process:
            rearranged_input = einops.rearrange(
                input,
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=self.patch_dim,
                p2=self.patch_dim,
            )

            assert rearranged_input.dtype == torch.half
            encoder_output = self.linear_encoder(rearranged_input)

            concatenated_tokens = encoder_output
            if self.class_token:
                cls_tokens = self.cls_token.expand(encoder_output.shape[0], -1, -1)
                concatenated_tokens = torch.cat((cls_tokens, encoder_output), dim=1)

            token_embeddings = concatenated_tokens + \
                    self.position_embeddings(self.position_ids[:, :concatenated_tokens.shape[1]])
            # [b, s, h] => [s, b, h]
            token_embeddings = token_embeddings.transpose(0, 1).contiguous()
            hidden_states = self.embedding_dropout(token_embeddings) 
            ##TODO: Possible Throughput Gains in the Future
            ##1. Could run preprocess first on rank 0, using torch.barrier(), try tensor_parallel for boradcasting
            ##2. Should I do dropout before or after sequence splitting? 
            ##3. Even better, I could split before linear encoding or before patchifying.
            ##NOTE: Priming DS Seq. Parallelism by splitting [s, b, h] -> [s/sp, b, h]

            ##TODO: (CRITICAL) Don't we need to use sequence_data_parallel instead of sequence_parallel_rank? However, sequence_data_parallel is just ddp? 
            # args = get_args()
            # print(f"rank: {args.rank}")
            # print(f"get_sequence_parallel_rank: {mpu.get_sequence_parallel_rank()}")
            # print(f"get_sequence_data_parallel_rank: {mpu.get_sequence_data_parallel_rank()}")
            
            # args = get_args()
            # print(f"args.rank: {args.rank}, hidden_states (before chunking): {hidden_states.shape}")
            if self.ds_sequence_parallel:
                seq_parallel_world_size = mpu.get_sequence_parallel_world_size()
                seq_parallel_world_rank = mpu.get_sequence_parallel_rank()
                # print(f"mpu.get_sequence_parallel_rank(): {mpu.get_sequence_parallel_rank()}")

                # assert self.seq_length % seq_parallel_world_size == 0
                ## This should terminate it, and if you want more seq_length...
                sub_seq_length = self.seq_length // seq_parallel_world_size
                sub_seq_start = seq_parallel_world_rank * sub_seq_length
                ##TODO: (CRITICAL) Last Rank might hold more tokens. Allows sequence parallelism with undivisable seq_length
                sub_seq_end = (seq_parallel_world_rank + 1) * sub_seq_length
                # if seq_parallel_world_rank == seq_parallel_world_size-1:
                #     print("hi")
                    # sub_seq_end = self.seq_length
                # else:
                #     print("hi")
                    # sub_seq_end = (seq_parallel_world_rank + 1) * sub_seq_length
                # torch.distributed.barrier()
                hidden_states = hidden_states[sub_seq_start:sub_seq_end, :, :] ## s, b, h
                # print(f"sub_seq_start, sub_seq_end: {sub_seq_start, sub_seq_end}")
                # print(f"seq_parallel_world_rank: {seq_parallel_world_rank}")
                # print(f"seq_parallel_world_rank: {seq_parallel_world_rank}")
                # print(f"preprocessed hidden_states.shape: {hidden_states.shape}")
            # print(f"args.rank: {args.rank}, hidden_states (after chunking): {hidden_states.shape}")
        else:
            hidden_states = input

        # with open("debug/output_SP.txt", "w") as f:
        #     if self.ds_sequence_parallel:
        #         f.write(f"first hidden_state: {gather_from_sequence_parallel_group(hidden_states)}")
        #         f.write(f"first hidden_state shape: {gather_from_sequence_parallel_group(hidden_states).shape}")
        #     else:
        #         f.write(f"first hidden_state: {hidden_states}")
        #         f.write(f"first hidden_state shape: {hidden_states.shape}")

        hidden_states = self.transformer(hidden_states, None)[0] ## [0] ignore moe losses

        with open("debug/output_SP.txt", "a") as f:
            if self.ds_sequence_parallel:
                # if seq_parallel_world_rank == 0:
                    ## just the first block
                f.write(f"seq_parallel_world_rank: {seq_parallel_world_rank}\n")
                # f.write(f"first hidden_state: {gather_from_sequence_parallel_group(hidden_states)}\n")
                # f.write(f"first hidden_state shape: {gather_from_sequence_parallel_group(hidden_states).shape}\n")
                f.write(f"first hidden_state: {hidden_states}\n")
                f.write(f"first hidden_state shape: {hidden_states.shape}\n")
            else:
                dp_rank = mpu.get_data_parallel_rank()
                f.write(f"dp_parallel_world_rank: {dp_rank}\n")
                f.write(f"first hidden_state: {hidden_states}\n")
                f.write(f"first hidden_state shape: {hidden_states.shape}\n")

        ##NOTE: This seems like a pointless thing to do because you are going to extract the first token only anyway.
        # from megatron.core import tensor_parallel
        args = get_args()
        # print(f"before rank: {args.rank}, hidden_states.shape: {hidden_states.shape}")
        hidden_states = gather_from_sequence_parallel_group(hidden_states) ## gather across seq dim (n/sp, b, h) -> (n, b, h)
        # print(f"after rank: {args.rank}, {hidden_states.shape}")
        # hidden_states = tensor_parallel.gather_from_sequence_parallel_region(hidden_states) ## gather across seq dim (n/sp, b, h) -> (n, b, h)
        # torch.distributed.breakpoint()
        # print(f"mpu.get_sequence_parallel_rank: {mpu.get_sequence_parallel_rank}")
        
        ## only extract the clf token in the end.
        # if mpu.get_sequence_parallel_rank != 0:
        #     return None

        if self.post_process:
            # [s b h] => [b s h]
            if self.single_token_output:
                hidden_states = hidden_states[0]
            else:
                hidden_states = hidden_states.transpose(0, 1).contiguous() ## Should always transpose back. 

        return hidden_states
    

def gather_from_sequence_parallel_group(input_):
    """Gather tensors and concatinate along the first dimension."""
    from megatron.core.parallel_state import get_sequence_parallel_group, get_sequence_parallel_world_size
    from deepspeed.accelerator import get_accelerator

    world_size = get_sequence_parallel_world_size()
    if world_size == 1:
        # Bypass the function if we are using only 1 GPU.
        return input_

    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size

    output = torch.empty(dim_size, dtype=input_.dtype,
                         device=get_accelerator().current_device_name())
    # torch.distributed._all_gather_base(output, input_.contiguous(),
    torch.distributed.all_gather_into_tensor(output, input_.contiguous(),
                                       group=get_sequence_parallel_group())
    return output