"""Classification head supporting multiple labelled tasks via a ModuleDict."""
import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """One linear classifier per task.

    Args:
        label_category_dict: Mapping ``{task_name: num_classes}``. For binary
            classification with BCE loss, use ``num_classes = 1``.
        in_channel: Input feature dimension.
        bias: Whether the linear layer uses bias.
    """

    def __init__(self, label_category_dict: dict, in_channel: int, bias: bool = True):
        super().__init__()
        self.heads = nn.ModuleDict({
            name: nn.Linear(in_channel, n_out, bias=bias)
            for name, n_out in label_category_dict.items()
        })

    def forward(self, f):
        if isinstance(f, dict):
            return {name: head(f[name]) for name, head in self.heads.items()}
        if isinstance(f, list):
            return {name: head(f[i]) for i, (name, head) in enumerate(self.heads.items())}
        return {name: head(f) for name, head in self.heads.items()}
