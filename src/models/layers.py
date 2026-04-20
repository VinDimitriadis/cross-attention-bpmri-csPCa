"""Common building blocks: pooling, normalization/activation factories, ConvNormActive."""
import torch.nn as nn


class GlobalAvgPool(nn.Module):
    """Global average pooling over spatial dims (works for 1D/2D/3D)."""

    def forward(self, x):
        if x.dim() == 2:
            return x
        if x.dim() == 3:
            return nn.functional.adaptive_avg_pool1d(x, 1).flatten(1)
        if x.dim() == 4:
            return nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        if x.dim() == 5:
            return nn.functional.adaptive_avg_pool3d(x, 1).flatten(1)
        raise ValueError(f"Unsupported tensor rank {x.dim()}.")


class GlobalMaxPool(nn.Module):
    """Global max pooling over spatial dims (works for 1D/2D/3D)."""

    def forward(self, x):
        if x.dim() == 2:
            return x
        if x.dim() == 3:
            return nn.functional.adaptive_max_pool1d(x, 1).flatten(1)
        if x.dim() == 4:
            return nn.functional.adaptive_max_pool2d(x, 1).flatten(1)
        if x.dim() == 5:
            return nn.functional.adaptive_max_pool3d(x, 1).flatten(1)
        raise ValueError(f"Unsupported tensor rank {x.dim()}.")


class GlobalMaxAvgPool(nn.Module):
    """Average of global-max and global-avg pooling."""

    def __init__(self):
        super().__init__()
        self.gap = GlobalAvgPool()
        self.gmp = GlobalMaxPool()

    def forward(self, x):
        return (self.gmp(x) + self.gap(x)) / 2.0


def make_norm(dim: int, channels: int, norm: str = "bn", gn_groups: int = 8) -> nn.Module:
    """Norm-layer factory. dim in {1,2,3}; norm in {'bn','in','gn','None',None}."""
    if norm == "bn":
        return {1: nn.BatchNorm1d, 2: nn.BatchNorm2d, 3: nn.BatchNorm3d}[dim](channels)
    if norm == "in":
        return {1: nn.InstanceNorm1d, 2: nn.InstanceNorm2d, 3: nn.InstanceNorm3d}[dim](channels)
    if norm == "gn":
        return nn.GroupNorm(gn_groups, channels)
    if norm in ("None", None):
        return nn.Identity()
    raise ValueError(f"Unknown norm: {norm}")


def make_active(active: str = "relu") -> nn.Module:
    """Activation factory: relu / leakyrelu / selu / gelu / None."""
    if active == "relu":
        return nn.ReLU(inplace=True)
    if active == "leakyrelu":
        return nn.LeakyReLU(inplace=True)
    if active == "selu":
        return nn.SELU(inplace=True)
    if active == "gelu":
        return nn.GELU()
    if active in ("None", None):
        return nn.Identity()
    raise ValueError(f"Unknown activation: {active}")


def make_conv(in_ch: int, out_ch: int, kernel_size: int,
              padding: int = 1, stride: int = 1, dim: int = 3, bias: bool = False) -> nn.Module:
    """Conv factory for 1D/2D/3D."""
    conv_cls = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[dim]
    return conv_cls(in_ch, out_ch, kernel_size, padding=padding, stride=stride, bias=bias)


class ConvNormActive(nn.Module):
    """Conv -> Norm -> Activation -> Dropout."""

    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3,
                 norm="bn", active="relu", gn_groups=8, dim=3, padding=1, dropout=0.5):
        super().__init__()
        self.conv = make_conv(in_channels, out_channels, kernel_size,
                              padding=padding, stride=stride, dim=dim)
        self.norm = make_norm(dim, out_channels, norm, gn_groups)
        self.active = make_active(active)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.active(self.norm(self.conv(x))))
