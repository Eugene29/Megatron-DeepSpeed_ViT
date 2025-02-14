from megatron.model.module import MegatronModule

import torch
import torch.distributed  as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
import einops
import argparse
from timm.layers import to_2tuple
# from timm.layers import DropPath, Mlp, ClassifierHead, to_2tuple, _assert

import logging
import math
from typing import Tuple, Optional, List, Union, Any, Type
from types import SimpleNamespace
from ruamel.yaml import YAML

def swin_from_yaml(fname, checkpoint_stages=False):
    yaml = YAML()
    with open(fname) as f:
        hparams = yaml.load(f)
    params = SimpleNamespace()
    for k,v in hparams.items():
        setattr(params, k, v)
    return swinv2net(params, checkpoint_stages=checkpoint_stages)

def swinv2net(params, checkpoint_stages=False):
    act_ckpt = checkpoint_stages or params.activation_ckpt
    return SwinTransformerV2Cr(
                  img_size=params.img_size,
                  patch_size=params.patch_size,
                  depths = (params.depth,),   
                  num_heads=(params.num_heads,),
                  in_chans=params.n_in_channels,
                  out_chans=params.n_out_channels,
                  embed_dim=params.embed_dim,
                  img_window_ratio=params.window_ratio,
                  drop_path_rate=params.drop_path_rate,
                  full_pos_embed=params.full_pos_embed,
                  rel_pos=params.rel_pos,
                  mlp_ratio=params.mlp_ratio,
                  checkpoint_stages=act_ckpt,
                  residual=params.residual
    )
 
def swin_flop_count(
    args: argparse.Namespace
) -> int:
    img_h, img_w = args.img_h, args.img_w
    p = args.patch_dim
    if args.swin_window_size is not None:
        assert isinstance(args.swin_window_size, int), \
            "haven't thought about non-square window size yet"
        L_w = args.swin_window_size**2  # seq length per window
    else:  ## Fall back to window2image ratio
        assert (img_h/args.swin_window2image_ratio) % p == 0 
        assert (img_w/args.swin_window2image_ratio) % p == 0 
        window_size_h = img_h / args.swin_window2image_ratio / p
        window_size_w = img_w / args.swin_window2image_ratio / p
        L_w = window_size_h * window_size_w
    B = args.global_batch_size
    l = args.num_layers
    N_w = img_h * img_w / L_w / p**2 # total num windows
    Bw = B * N_w  # num total windows
    c = args.num_channels
    d = args.hidden_size

    pre_and_post_process = 2 * Bw * p**2 * c * d
    QKVO = 4 * Bw * L_w * d**2
    FA = 2 * Bw * L_w**2 * d
    MLP = 2 * Bw * L_w * args.hidden_size * args.ffn_hidden_size
    fwd_flop = (QKVO+FA+MLP)*l + pre_and_post_process
    return 6 * fwd_flop  # 3x for fwd+bwd, 2x for mult+add operations

def window_partition(x, window_size):
    r"""Partition image (B, h, w, d) into sequence of windows or patches 
    (B*s, w_h, w_w, C)

    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    window_h, window_w = window_size
    x = x.view(
        B, H // window_h, window_h, W // window_w, window_w, C
    )  # [B, N_h, N_w, w_h, w_w, C]
    windows = x.permute(0, 1, 3, 2, 4, 5).reshape(
        -1, window_h, window_w, C
    )  # [B*s, w_h, w_w, C]
    return windows

def window_partition_reverse(windows, window_size, img_size):
    r"""Reverse the window partitioning (B*s, W_h, W_w, C) -> (B, H, W, C)
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    H, W = img_size
    window_h, window_w = window_size
    B = int(windows.shape[0] / (H * W / window_h / window_w))
    x = windows.view(B, H // window_h, W // window_w, window_h, window_w, -1)
    # [B, N_h, N_w, w_h, w_w, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    # [B, H, W, C]
    return x


class Mlp(nn.Module):
    def __init__(
        self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    

class WindowMultiHeadAttentionNoPos(MegatronModule):
    r"""This class implements window-based Multi-Head-Attention with log-spaced continuous position bias.

    Args:
        dim (int): Number of input features
        window_size (int): Window size
        num_heads (int): Number of attention heads
        drop_attn (float): Dropout rate of attention map
        drop_proj (float): Dropout rate after projection
        meta_hidden_dim (int): Number of hidden features in the two layer MLP meta network
        sequential_attn (bool): If true sequential self-attention is performed
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Tuple[int, int],
        drop_attn: float = 0.0,
        drop_proj: float = 0.0,
        sequential_attn: bool = False,
    ) -> None:
        super(WindowMultiHeadAttentionNoPos, self).__init__()
        assert dim % num_heads == 0, \
            "The number of input features (in_features) are not divisible by the number of heads (num_heads)."
        self.in_features: int = dim
        self.window_size: Tuple[int, int] = window_size
        self.num_heads: int = num_heads
        self.sequential_attn: bool = sequential_attn

        self.qkv = nn.Linear(in_features=dim, out_features=3*dim, bias=True)
        self.attn_drop = nn.Dropout(drop_attn)
        self.proj = nn.Linear(in_features=dim, out_features=dim, bias=True)
        self.proj_drop = nn.Dropout(drop_proj)
        # # NOTE old checkpoints used inverse of logit_scale ('tau') following the paper, see conversion fn
        # self.logit_scale = nn.Parameter(torch.log(10 * torch.ones(num_heads)))

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """ Forward pass.
        Args:
            x (torch.Tensor): Input tensor of the shape (B * windows, L, C)
            mask (Optional[torch.Tensor]): Attention mask for the shift case

        Returns:
            Output tensor of the shape [B * windows, L, C]
        """
        Bw, L, C = x.shape
        qkv = self.qkv(x).view(
            Bw, L, 3, self.num_heads, C // self.num_heads
        ).permute(2, 0, 3, 1, 4)  # Bw, L, C -> 3, Bw, hc, L, hs
        query, key, value = [x.contiguous() for x in qkv.unbind(0)]

        # Flash Attention  
        # TODO: Is cosine attention in our interest or is normalizing query and
        # key before attention good enough? 
        query_normed, key_normed = query, key
        x = F.scaled_dot_product_attention(
            query_normed, key_normed, value 
        ).transpose(1, 2).reshape(Bw, L, C)  # Bw, hc, L, hs -> # Bw, L, h

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerV2CrBlock(MegatronModule):
    r"""This class implements the Swin transformer block.

    Args:
        dim (int): Number of input channels
        num_heads (int): Number of attention heads to be utilized
        feat_size (Tuple[int, int]): Input resolution
        window_size (Tuple[int, int]): Window size to be utilized
        shift_size (int): Shifting size to be used
        mlp_ratio (int): Ratio of the hidden dimension in the FFN to the input channels
        proj_drop (float): Dropout in input mapping
        drop_attn (float): Dropout rate of attention map
        drop_path (float): Dropout in main path
        extra_norm (bool): Insert extra norm on 'main' branch if True
        sequential_attn (bool): If true sequential self-attention is performed
        norm_layer (Type[MegatronModule]): Type of normalization layer to be utilized
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        feat_size: Tuple[int, int],
        window_size: Tuple[int, int],
        shift_size: Tuple[int, int] = (0, 0),
        mlp_ratio: float = 4.0,
        init_values: Optional[float] = 0,
        proj_drop: float = 0.0,
        drop_attn: float = 0.0,
        drop_path: float = 0.0,
        extra_norm: bool = False,
        sequential_attn: bool = False,
        norm_layer: Type[MegatronModule] = nn.LayerNorm,
        rel_pos: bool = True,
    ) -> None:
        super(SwinTransformerV2CrBlock, self).__init__()
        self.dim: int = dim
        self.feat_size: Tuple[int, int] = feat_size
        self.target_shift_size: Tuple[int, int] = to_2tuple(shift_size)
        self.window_size, self.shift_size = self._calc_window_shift(to_2tuple(window_size))
        assert self.window_size == window_size, "likely undesired side-effect"
        assert self.shift_size == shift_size, "likely undesired side-effect"
        self.window_area = self.window_size[0] * self.window_size[1]
        self.init_values: Optional[float] = init_values

        # attn branch
        self.attn = WindowMultiHeadAttentionNoPos(
            dim=dim,
            num_heads=num_heads,
            window_size=self.window_size,
            drop_attn=drop_attn,
            drop_proj=proj_drop,
            sequential_attn=sequential_attn,
        )
        self.norm1 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            drop=proj_drop,
            out_features=dim,
        )
        self.norm2 = norm_layer(dim)
        self.init_weights()

    def _calc_window_shift(self, target_window_size):
        ## window_size = window_size if window_size < input resolution
        window_size = [f if f <= w else w for f, w in zip(self.feat_size, target_window_size)]
        ## shift_size = target_shift_size if window_size <= input resolution otherwise 0
        shift_size = [0 if f <= w else s for f, w, s in zip(self.feat_size, window_size, self.target_shift_size)]
        return tuple(window_size), tuple(shift_size)

    def init_weights(self):
        # extra, module specific weight init
        if self.init_values is not None:
            nn.init.constant_(self.norm1.weight, self.init_values)
            nn.init.constant_(self.norm2.weight, self.init_values)

    def _shifted_window_attn(self, x):
        B, h, w, d = x.shape
        
        # cyclic shift (shift once every layer without reversing) [B, h, w, d]
        sh, sw = self.shift_size
        assert isinstance(sh, int), isinstance(sw, int)
        do_shift: bool = any(self.shift_size)
        if do_shift:
            x = torch.roll(x, shifts=(-sh, -sw), dims=(1, 2))  # [B, h, w, d]
        else:
            print("Shifting is not happening. There may be a logic issue")

        # partition windows:  [Bw, w_h*w_w, d]
        ## TODO: move window parition in side do_shift?
        x_windows = window_partition(x, self.window_size)  # Bw, w_h, w_w, d
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1], d)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows)  # [Bw, L, d]

        # merge windows: [B, h, w, d]
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], d)
        x = window_partition_reverse(attn_windows, self.window_size, self.feat_size)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.norm1(self._shifted_window_attn(x))  # [B, h, w, d]
        x = x + self.norm2(self.mlp(x))  # [B, h, w, d]
        return x


class SwinTransformerV2CrStage(MegatronModule):
    r"""This class implements a stage of the Swin transformer including multiple layers.

    Args:
        embed_dim (int): Number of input channels
        depth (int): Depth of the stage (number of layers)
        downscale (bool): If true input is downsampled (see Fig. 3 or V1 paper)
        feat_size (Tuple[int, int]): input feature map size (H, W)
        num_heads (int): Number of attention heads to be utilized
        window_size (int): Window size to be utilized
        mlp_ratio (int): Ratio of the hidden dimension in the FFN to the input channels
        proj_drop (float): Dropout in input mapping
        drop_attn (float): Dropout rate of attention map
        drop_path (float): Dropout in main path
        norm_layer (Type[MegatronModule]): Type of normalization layer to be utilized. Default: nn.LayerNorm
        extra_norm_period (int): Insert extra norm layer on main branch every N (period) blocks
        extra_norm_stage (bool): End each stage with an extra norm layer in main branch
        sequential_attn (bool): If true sequential self-attention is performed
    """

    def __init__(
        self,
        embed_dim: int,
        depth: int,
        downscale: bool,
        num_heads: int,
        feat_size: Tuple[int, int],
        window_size: Tuple[int, int],
        mlp_ratio: float = 4.0,
        init_values: Optional[float] = 0.0,
        proj_drop: float = 0.0,
        drop_attn: float = 0.0,
        drop_path: Union[List[float], float] = 0.0,
        norm_layer: Type[MegatronModule] = nn.LayerNorm,
        extra_norm_period: int = 0,
        extra_norm_stage: bool = False,
        sequential_attn: bool = False,
        rel_pos: bool = True,
        grad_checkpointing: bool = False,
    ) -> None:
        super(SwinTransformerV2CrStage, self).__init__()
        self.downscale: bool = downscale
        self.feat_size: Tuple[int, int] = (feat_size[0] // 2, feat_size[1] // 2) if downscale else feat_size
        self.grad_checkpointing = grad_checkpointing

        ## TODO: how to remove this cleanly? 
        def _extra_norm(index):
            i = index + 1
            if extra_norm_period and i % extra_norm_period == 0:
                return True
            return i == depth if extra_norm_stage else False

        # shift every layer instead of twice every other layer
        shift_size = (window_size[0] // 2, window_size[1] // 2)
        self.blocks = nn.Sequential(*[
            SwinTransformerV2CrBlock(
                dim=embed_dim,
                num_heads=num_heads,
                feat_size=self.feat_size,
                window_size=window_size,
                shift_size=shift_size,
                mlp_ratio=mlp_ratio,
                init_values=init_values,
                proj_drop=proj_drop,
                drop_attn=drop_attn,
                drop_path=drop_path[index] if isinstance(drop_path, list) else drop_path,
                extra_norm=_extra_norm(index),
                sequential_attn=sequential_attn,
                norm_layer=norm_layer,
                rel_pos=rel_pos,
            )
            for index in range(depth)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        Args:
            x (torch.Tensor): Input tensor of the shape [B, C, H, W] or [B, L, C]
        Returns:
            output (torch.Tensor): Output tensor of the shape [B, 2 * C, H // 2, W // 2]
        """
        for block in self.blocks:
            # Perform checkpointing if utilized
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return x

class SwinTransformerV2Cr(MegatronModule):
    r""" Swin Transformer V2
        A PyTorch impl of : `Swin Transformer V2: Scaling Up Capacity and Resolution`  -
          https://arxiv.org/pdf/2111.09883

    Args:
        img_size: Input resolution.
        window_size: Window size. If None, img_size // window_div
        img_window_ratio: Window size to image size ratio.
        patch_size: Patch size.
        in_chans: Number of input channels.
        depths: Depth of the stage (number of layers).
        num_heads: Number of attention heads to be utilized.
        embed_dim: Patch embedding dimension.
        num_classes: Number of output classes.
        mlp_ratio:  Ratio of the hidden dimension in the FFN to the input channels.
        drop_rate: Dropout rate.
        proj_drop_rate: Projection dropout rate.
        attn_drop_rate: Dropout rate of attention map.
        drop_path_rate: Stochastic depth rate.
        norm_layer: Type of normalization layer to be utilized.
        extra_norm_period: Insert extra norm layer on main branch every N (period) blocks in stage
        extra_norm_stage: End each stage with an extra norm layer in main branch
        sequential_attn: If true sequential self-attention is performed.
    """

    def __init__(
        self,
        config=None,  # TransformerConfig? 
        img_size: Tuple[int, int] = (224, 224),
        patch_size: int = 4,
        window_size: Optional[int] = None,
        img_window_ratio: int = 32,
        in_chans: int = 3,
        out_chans: int = 3,
        embed_dim: int = 96,
        depths: Tuple[int, ...] = (2, 2, 6, 2),
        num_heads: Tuple[int, ...] = (3, 6, 12, 24),
        mlp_ratio: float = 4.0,
        init_values: Optional[float] = 0.,
        drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        extra_norm_period: int = 0,
        extra_norm_stage: bool = False,
        sequential_attn: bool = False,
        global_pool: str = 'avg',
        weight_init='skip',
        full_pos_embed: bool = False,
        rel_pos: bool = True,
        checkpoint_stages: bool = False,
        residual:  bool = False,
        **kwargs: Any
    ) -> None:
        super(SwinTransformerV2Cr, self).__init__(config=config)  # initiate Megatron config
        img_size = to_2tuple(img_size)
        if window_size is not None:
            window_size = to_2tuple(window_size)
        else:
            window_size = tuple([s // img_window_ratio for s in img_size])

        self.patch_size: int = patch_size
        self.img_size: Tuple[int, int] = img_size
        self.window_size: Tuple[int, int] = window_size
        self.num_features: int = int(embed_dim)
        self.out_chans: int = out_chans
        self.full_pos_embed: bool = full_pos_embed
        self.checkpoint_stages = checkpoint_stages
        self.residual = residual
        self.depth = len(depths)
        self.patch_grid_size: Tuple[int, int] = (
            img_size[0]//patch_size, img_size[1]//patch_size
        )
        self.window_area = self.window_size[0] * self.window_size[1]

        print(f"window_size: {self.window_size}", flush=True)
        print(f"img_size: {img_size}", flush=True)
        print(f"sequence length per window: {self.window_area}")
        print("total sequence length per MBS: {}".format(
            self.patch_grid_size[0] * self.patch_grid_size[1]
        ))
        assert patch_size * window_size[0] <= img_size[0], \
            "window height cannot be bigger than the image height"
        assert patch_size * window_size[1] <= img_size[1], \
            "window width cannot be bigger than the image width"

        ## Patchify
        self.patchify_proj = nn.Linear(in_chans * self.patch_size**2, embed_dim)
        self.post_patchify_norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        assert norm_layer is not None

        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        stages = []
        for stage_idx, (depth, num_heads) in enumerate(zip(depths, num_heads)):
            stages += [SwinTransformerV2CrStage(
                embed_dim=embed_dim,
                depth=depth,
                downscale=False,
                feat_size=(
                    self.patch_grid_size[0],
                    self.patch_grid_size[1]
                ),
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                init_values=init_values,
                proj_drop=proj_drop_rate,
                drop_attn=attn_drop_rate,
                drop_path=dpr[stage_idx],
                extra_norm_period=extra_norm_period,
                extra_norm_stage=extra_norm_stage or (stage_idx + 1) == len(depths),  # last stage ends w/ norm
                sequential_attn=sequential_attn,
                norm_layer=norm_layer,
                rel_pos=rel_pos,
                grad_checkpointing=self.checkpoint_stages,
            )]

        self.stages = nn.Sequential(*stages)
        self.head = nn.Linear(embed_dim, self.out_chans * self.patch_size**2, bias=False)

        if self.full_pos_embed:
            ## TODO: Replace with ROPE
            self.pos_embed = nn.Parameter(torch.randn(1, *self.patch_grid_size, embed_dim) * .02)

        # # current weight init skips custom init and uses pytorch layer defaults, seems to work well
        # # FIXME more experiments needed
        # if weight_init != 'skip':
        #     named_apply(init_weights, self)

    def set_input_tensor(self, input_tensor):
        """place holder for megatron.model.transformer.set_input_tensor()"""
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ## Preprocess (patchification)
        # h: num patches row-wise
        # c: num patches col-wise
        x = einops.rearrange(x, 
            "B c (h p1) (w p2) -> B h w p1 p2 c",
            p1=self.patch_size,
            p2=self.patch_size
        )

        x = x.flatten(start_dim=-3)  ## [B, h, w, p*p*c]
        x = self.patchify_proj(x)  ## [B, h, w, d]
        x = self.post_patchify_norm(x)  ## [B, h, w, d]

        ## Backbone
        if self.full_pos_embed:
            x = x + self.pos_embed
        x = self.stages(x)

        ## Head
        x = self.head(x)  # [B, h, w, d]

        ## TODO: post-processing: upsampling with linear + pixel_shuffle + conv x n
        # print(f"torch.cuda.memory_summary(): {torch.cuda.memory_summary()}")
        x = einops.rearrange(x, 
            "B h w (p1 p2 c) -> B c (h p1) (w p2)",
            p1=self.patch_size,
            p2=self.patch_size,
            c=self.out_chans
        )
        return x

if __name__ == "__main__":
    DEVICE = "xpu"
    # DEVICE = "cpu"
    IMG_SIZE=32
    DTYPE = torch.float32
    model = SwinTransformerV2Cr(
        img_size=[IMG_SIZE, IMG_SIZE],
        patch_size=4,
        depths=[1],
        num_heads=(1,),
        in_chans=3,
        out_chans=3,
        embed_dim=8,
        img_window_ratio=8,  # Modify number of windows
        drop_path_rate=0,  # Stochastic Depth
        full_pos_embed=True,  # TODO: Replace with ROPE?
        rel_pos=False,  # TODO: REMOVE from args?
        mlp_ratio=1,  # Fixed projection dimension
        checkpoint_stages=False,  # TODO: Enable activation checkpointing
        residual=False,  # TODO: What is residual doing?
    ).to(DEVICE)
    B, C, H, W = [1, 3, IMG_SIZE, IMG_SIZE]
    x = torch.randn(B, C, H, W, device=DEVICE, dtype=DTYPE)
    # print(f"model: {model}", flush=True)
    # print(f"x.shape: {x.shape}", flush=True)
    model(x)