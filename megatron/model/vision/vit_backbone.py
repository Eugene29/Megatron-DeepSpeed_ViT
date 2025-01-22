# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Vision Transformer(VIT) model."""

import math
import einops
import torch
# import apex
import os
import torch.distributed
import torch.nn.functional as F
from megatron import get_args
from megatron.model.transformer import ParallelTransformer
from megatron.model.utils import (
    # get_linear_layer,
    init_method_normal,
    # scaled_init_method_normal,
)
from deepspeed import comm as dist
from megatron.core import parallel_state as mpu
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

    def __init__(self, hidden_size, num_classes, config):
        super(VitMlpHead, self).__init__()
        self.dense_in = torch.nn.Linear(hidden_size, hidden_size)
        self.dense_out = torch.nn.Linear(hidden_size, num_classes)
        torch.nn.init.constant_(self.dense_out.bias, -10)

        ## TODO: Enable TP (But below might be useless if VIT is not an autoencoder)
        # from megatron.core.tensor_parallel import ColumnParallelLinear
        # self.dense_in = ColumnParallelLinear(hidden_size, hidden_size, config=config, init_method=config.init_method, gather_output=True, forSP=True)
        # self.dense_out = ColumnParallelLinear(hidden_size, num_classes, config=config, init_method=config.init_method, gather_output=False, forSP=True)

    def forward(self, hidden_states):
        # hidden_states: [b, 1, h]
        # sequence_index: index of the token to pool.
        dense_in_result = self.dense_in(hidden_states)
        tanh_result = torch.tanh(dense_in_result)
        dense_out_result = self.dense_out(tanh_result)

        ## TODO: Enable TP (But below might be useless if VIT is not an autoencoder)
        # dense_in_result, _ = self.dense_in(hidden_states)
        # dense_out_result, _ = self.dense_out(tanh_result)

        return dense_out_result


def isPerfectSquare(x):
    if(x >= 0):
        sr = math.sqrt(x)
        return (int(sr) * int(sr) == x)
    return False


## Q. What is this doing? Hooks? 
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
                input_param_grid, scale_factor=scale_factor, mode="bilinear" ## Q. What does bilinear mean again? 
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

        self.flatten_dim = self.patch_dim * self.patch_dim * args.num_channels
        self.input_tensor = None
        self.position_ids = None

        if "VIT3D" not in os.environ:
            self.num_patches_per_dim_h = self.img_h // self.patch_dim
            self.num_patches_per_dim_w = self.img_w // self.patch_dim
            self.num_patches = self.num_patches_per_dim_h * self.num_patches_per_dim_w
            self.seq_length = self.num_patches + (CLASS_TOKEN_LENGTH if self.class_token else 0) ## why are you recalcualting seq length if it is required as env variable? 
        else:
            img_d = int(os.environ["IMG_D"]) ## image depth
            assert img_d % self.patch_dim == 0, f"img_d:, {img_d}, patch_dim: {self.patch_dim}"
            self.flatten_dim = self.patch_dim * self.patch_dim * self.patch_dim * args.num_channels
            self.seq_length = int(os.environ["SEQ_LEN"])

        if self.ds_sequence_parallel:
            sp = mpu.get_sequence_parallel_world_size()
            if not self.seq_length % sp == 0:
                print("beaware, your sequence is uneven")
            if not (args.num_attention_heads % sp == 0 and args.num_attention_heads > sp):
                print("beaware, your head count is invalid (i.e. (head count indivisible by SP) or (SP > head count)")
            # assert self.seq_length % sp == 0
            # assert args.num_attention_heads % sp == 0, "Num head is the max sp degree for Ulysses"


        if self.pre_process:
            # cls_token
            if self.class_token:
                self.cls_token = torch.nn.Parameter(
                    torch.randn(1, CLASS_TOKEN_LENGTH, self.hidden_size)
                )
                torch.nn.init.zeros_(self.cls_token)
            self.position_ids = torch.arange(self.seq_length, device=dist.get_local_rank()).expand(1, -1)
            
            # Linear encoder
            self.linear_encoder = torch.nn.Linear(
                self.flatten_dim, self.hidden_size
            )

            # Embedding
            self.pos_encoding = args.pos_encoding
            if self.pos_encoding:
                ## TODO: Understanding the arguments behind pos encoding and optimize it for longer sequences. 
                def positionalencoding1d(d_model, length, dev):
                    """
                    :param d_model: dimension of the model
                    :param length: length of positions
                    :return: length*d_model position matrix
                    """
                    if d_model % 2 != 0:
                        raise ValueError("Cannot use sin/cos positional encoding with "
                                        "odd dim (got dim={:d})".format(d_model))
                    dtype = torch.float16
                    pe = torch.zeros((length, d_model), dtype=dtype, device=dev)
                    position = torch.arange(0, length, dtype=dtype, device=dev).unsqueeze(1)
                    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=dtype, device=dev) *
                                        -(math.log(10000.0) / d_model)))
                    pe[:, 0::2] = torch.sin(position.float() * div_term)
                    pe[:, 1::2] = torch.cos(position.float() * div_term)
                    return pe

                pos_encoding = positionalencoding1d(self.hidden_size, self.seq_length, dist.get_local_rank())

            else:
                self.position_embeddings = torch.nn.Embedding(
                    self.seq_length, self.hidden_size
                )

            ## TODO: Mem fpt: Only generate relevant sequence's positional encoding from the get go
            sp = mpu.get_sequence_parallel_world_size()
            sp_rank = mpu.get_sequence_parallel_rank()
            sub_seq_length = self.seq_length // sp
            self.remainder_seq_len = self.seq_length % sp

            ## For pos_encoding, sub_seq_idx need to include clf_token
            ## For the actual sequence, sub_seq_idx need to eclude clf_token 
            ## Therefore, reduce the sub_sequence of first rank by 1 for clf token
            ## Cleaner code: reduce redundancy of code below
            if self.remainder_seq_len == 0:
                sub_seq_start = sp_rank * sub_seq_length
                sub_seq_end = (sp_rank + 1) * sub_seq_length
            else:
                seq_shard_list = [sub_seq_length+1] * self.remainder_seq_len + [sub_seq_length] * (sp-self.remainder_seq_len)
                sub_seq_start = sum(seq_shard_list[:sp_rank])
                sub_seq_end = sum(seq_shard_list[:sp_rank+1])
                print("likely will use uneven sequence parallelism")
            self.position_embeddings = pos_encoding[sub_seq_start:sub_seq_end, :] ## s, h ?

            if self.ds_sequence_parallel:
                if class_token:
                    self.sub_seq_start = sub_seq_start if sp_rank == 0 else sub_seq_start - 1
                    self.sub_seq_end = sub_seq_end - 1
                else:
                    self.sub_seq_start = sub_seq_start
                    self.sub_seq_end = sub_seq_end

            args.class_token_present = self.class_token
            
            if not self.pos_encoding:
                init_method_normal(args.init_method_std)(
                    self.position_embeddings.weight
                )
                self.position_embeddings._register_load_state_dict_pre_hook(
                    twod_interpolate_position_embeddings_hook
                )

            self.embedding_dropout = torch.nn.Dropout(args.hidden_dropout)
            # Dropout.
            # if self.sequence_parallel:
            #     # already partition sequence, do not need scatter_to_sequence_parallel_region
            #     # embeddings = tensor_parallel.scatter_to_sequence_parallel_region(embeddings)
            #     # already partition sequence, do not need scatter_to_sequence_parallel_region ?
            #     embeddings = tensor_parallel.scatter_to_sequence_parallel_region(embeddings)
            #     with tensor_parallel.get_cuda_rng_tracker().fork():
            #         embeddings = self.embedding_dropout(embeddings)

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
            if "VIT3D" not in os.environ:
                rearranged_input = einops.rearrange(
                    input,
                    "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                    p1=self.patch_dim,
                    p2=self.patch_dim,
                )
            else:
                rearranged_input = einops.rearrange(
                    input,
                    "b c (h p1) (w p2) (d p3) -> b (h w d) (p1 p2 p3 c)",
                    p1=self.patch_dim,
                    p2=self.patch_dim,
                    p3=self.patch_dim,
                )

            seq_parallel_rank = mpu.get_sequence_parallel_rank()
            if self.ds_sequence_parallel:
                rearranged_input = rearranged_input[:, self.sub_seq_start:self.sub_seq_end, :] ## b, s, h
                # print(f"rearranged_input.shape: {rearranged_input.shape}") ## TODO: how did uneven sequence parallelism work beforehand? 
                ## Q. Don't we need to use sequence_data_parallel instead of sequence_parallel_rank?
                ## > No, sequence_data_parallel_rank is the rank of both sp + dp groups. 
            # raise KeyboardInterrupt()

            # assert rearranged_input.dtype == torch.half ## Q. We should be able to use bf16 if we want? 
            encoder_output = self.linear_encoder(rearranged_input)

            ## TODO: think of a way to reduce the number of bools
            if self.class_token and seq_parallel_rank == 0:
                cls_tokens = self.cls_token.expand(encoder_output.shape[0], -1, -1)
                concatenated_tokens = torch.cat((cls_tokens, encoder_output), dim=1)
            else:
                concatenated_tokens = encoder_output

            if self.pos_encoding:
                token_embeddings = concatenated_tokens + self.position_embeddings
            else:
                ## TODO: When supporting undivisible SP, include clf token in pos encoding while both parallelizing sequence and pos embedding. 
                token_embeddings = concatenated_tokens + \
                        self.position_embeddings(self.position_ids[:, :concatenated_tokens.shape[1]])
            
            hidden_states = token_embeddings.transpose(0, 1).contiguous() # [b, s, h] => [s, b, h]
            
            hidden_states = self.embedding_dropout(hidden_states) 
        else:
            hidden_states = input

        # if torch.distributed.get_rank() == 0:
        #     hidden_states = hidden_states[:-1]
        #     print(f"hidden_states.shape: {hidden_states.shape}")
        # debug_mode = "DEBUG_FNAME" in os.environ
        # if debug_mode:
        #     debug_fname = os.environ["DEBUG_FNAME"]
        #     with open(debug_fname, "a") as f:
        #         f.write(f"\n[{mpu.get_sequence_parallel_rank()}] Before Transformer Layers: {hidden_states}\n")
        #         f.write(f"\n[{mpu.get_sequence_parallel_rank()}] Before Transformer Layers shape: {hidden_states.shape}\n")

        hidden_states = self.transformer(hidden_states, None)[0] ## [0] ignore moe losses

        if self.post_process:
            # [s b h] => [b s h]
            if self.single_token_output:
                hidden_states = hidden_states[0]
            else:
                if "GLOBAL_MEAN_POOLING" in os.environ:
                    print(f"hidden_states: {hidden_states.shape}")
                    hidden_states = torch.mean(hidden_states, dim=0)
                else:
                    ## we are using global mean pool otherwise for now. 
                    ## if not single token, then output the entire embedding. 
                    hidden_states = hidden_states.transpose(0, 1).contiguous()

        # if debug_mode:
        #     with open(debug_fname, "a") as f:
        #         f.write(f"\n [{mpu.get_sequence_parallel_rank()}] First token before MLP_head: {hidden_states}\n")
        #         f.write(f"\n [{mpu.get_sequence_parallel_rank()}] First token before MLP_head shape: {hidden_states.shape}\n")

        return hidden_states