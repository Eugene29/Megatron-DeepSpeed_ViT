# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Vision Transformer(VIT) model."""

import torch
import torch.distributed
from torch.nn.init import trunc_normal_
from megatron import get_args
from megatron.model.utils import get_linear_layer
from megatron.model.vision.vit_backbone import VitBackbone, VitMlpHead
from megatron.model.vision.mit_backbone import mit_b3_avg
from megatron.model.module import MegatronModule
import os

class VitClassificationModel(MegatronModule):
    """Vision Transformer Model."""

    def __init__(self, config, num_classes, finetune=False,
                 pre_process=True, post_process=True):
        super(VitClassificationModel, self).__init__(config=config) ## add config to the model.
        args = get_args()

        self.hidden_size = args.hidden_size
        self.num_classes = num_classes
        self.finetune = finetune
        self.pre_process = pre_process
        self.post_process = post_process
        self.backbone = VitBackbone(
            config=config,
            pre_process=self.pre_process,
            post_process=self.post_process,
            class_token=("GLOBAL_MEAN_POOLING" not in os.environ),
            single_token_output=True
        )
        
        if self.post_process:
            if not self.finetune:
                self.head = VitMlpHead(self.hidden_size, self.num_classes, config=config)
            else:
                self.head = get_linear_layer(
                    self.hidden_size,
                    self.num_classes,
                    torch.nn.init.zeros_,
                    gather_params_on_init=args.zero_stage == 3
                )

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        self.backbone.set_input_tensor(input_tensor)

    def forward(self, input):
        hidden_states = self.backbone(input)

        ## Only rank==0 has hidden_states
        # if self.post_process and hidden_states is not None:
        # from megatron.core.parallel_state import get_sequence_parallel_rank as get_seq_rank
        # seq_rank = get_seq_rank()
        # if self.post_process and seq_rank == 0:
        if self.post_process:
            hidden_states = self.head(hidden_states)
        # else:
        #     hidden_states = 0

        # import os
        # debug_fname = os.environ['DEBUG_FNAME']
        # if torch.distributed.get_rank()==0:
        #     with open(debug_fname, "a") as f:
        #         f.write(f"Final output: {hidden_states}\n")
        #         f.write(f"Final output shape: {hidden_states.shape}\n")
        return hidden_states


class MitClassificationModel(MegatronModule):
    """Mix vision Transformer Model."""

    def __init__(self, num_classes,
                 pre_process=True, post_process=True):
        super(MitClassificationModel, self).__init__()
        args = get_args()

        self.hidden_size = args.hidden_size
        self.num_classes = num_classes

        self.backbone = mit_b3_avg()
        self.head = torch.nn.Linear(512, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, torch.nn.Linear) and m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        pass

    def forward(self, input):
        hidden_states = self.backbone(input)
        hidden_states = self.head(hidden_states)

        return hidden_states
