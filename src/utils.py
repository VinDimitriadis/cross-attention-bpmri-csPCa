"""General utilities: reproducibility and tensor resizing."""
import os
import random
import numpy as np
import torch
import torch.nn.functional as F


def seed_everything(seed: int = 42) -> None:
    """Seed python, numpy and torch for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resize_tensor(x: torch.Tensor, scale_factor=None, size=None) -> torch.Tensor:
    """Dimension-aware interpolation for 1D / 2D / 3D tensors."""
    if x.dim() == 3:
        return F.interpolate(x, scale_factor=scale_factor, size=size,
                             mode="linear", align_corners=True)
    if x.dim() == 4:
        return F.interpolate(x, scale_factor=scale_factor, size=size,
                             mode="bicubic", align_corners=True)
    if x.dim() == 5:
        return F.interpolate(x, scale_factor=scale_factor, size=size,
                             mode="trilinear", align_corners=True)
    raise ValueError(f"Unsupported tensor rank {x.dim()} for resize_tensor.")
