"""3D VGG-style encoder used per sequence (T2W / DWI / ADC)."""
import torch.nn as nn

from .layers import ConvNormActive
from ..utils import resize_tensor


class VGGBlock(ConvNormActive):
    """Single Conv-Norm-Active-Dropout block (alias of ConvNormActive)."""


class VGGStage(nn.Module):
    """Stack of `block_num` VGG blocks at a given channel width."""

    def __init__(self, in_channels, out_channels, block_num=2,
                 norm="bn", active="relu", gn_groups=8, dim=3):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(block_num):
            ic = in_channels if i == 0 else out_channels
            self.blocks.append(
                VGGBlock(ic, out_channels, norm=norm, active=active,
                         gn_groups=gn_groups, dim=dim)
            )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class VGGEncoder(nn.Module):
    """VGG-style encoder with configurable depth/width and per-stage downsampling.

    For each stage the input is (a) processed by a VGGStage and (b) downsampled
    by `downsample_ratio[stage]`. When `downsample_first=True` the order is flipped.
    """

    def __init__(self,
                 in_channels: int,
                 stage_output_channels=(64, 128, 256, 512),
                 blocks=(1, 2, 3, 4),
                 downsample_ratio=(0.5, 0.5, 0.5, 0.5),
                 downsample_first: bool = False,
                 norm: str = "bn",
                 active: str = "relu",
                 gn_groups: int = 8,
                 dim: int = 3):
        super().__init__()
        assert len(stage_output_channels) == len(blocks) == len(downsample_ratio)

        self.downsample_ratio = downsample_ratio
        self.downsample_first = downsample_first

        self.stages = nn.ModuleList()
        for i, out_ch in enumerate(stage_output_channels):
            ic = in_channels if i == 0 else stage_output_channels[i - 1]
            self.stages.append(
                VGGStage(ic, out_ch, block_num=blocks[i],
                         norm=norm, active=active, gn_groups=gn_groups, dim=dim)
            )

    def forward(self, x):
        features = []
        for stage, ratio in zip(self.stages, self.downsample_ratio):
            if self.downsample_first:
                x = resize_tensor(x, scale_factor=ratio)
                x = stage(x)
            else:
                x = stage(x)
                x = resize_tensor(x, scale_factor=ratio)
            features.append(x)
        return features
