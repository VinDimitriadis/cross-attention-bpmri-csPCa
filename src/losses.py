"""Loss functions and regularizers used for training."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def focal_bce_loss(logits: torch.Tensor,
                   targets: torch.Tensor,
                   class_weights: torch.Tensor,
                   gamma: float = 0.0,
                   reduction: str = "mean") -> torch.Tensor:
    """Binary cross-entropy with a focal modulating factor and per-class weights.

    The configuration used in the paper is gamma = 2 with class-balanced weights [1.755, 0.699]. 
    Plain BCE is recovered with gamma = 0 and class_weights = [1.0, 1.0].

    Args:
        logits: Raw scores, shape ``(B, 1)``.
        targets: Binary targets ``{0, 1}``, shape ``(B, 1)``.
        class_weights: Tensor ``[w_neg, w_pos]`` on the same device as ``logits``.
        gamma: Focal exponent.
        reduction: ``'mean'`` or ``'none'``.

    Class weights can be computed as::

        num_neg = (y == 0).sum()
        num_pos = (y == 1).sum()
        n = num_neg + num_pos
        w_neg = n / (2 * num_neg)
        w_pos = n / (2 * num_pos)
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    pt = torch.where(targets == 1, probs, 1 - probs)
    focal_weight = (1 - pt).pow(gamma)
    cw = torch.where(targets == 1, class_weights[1], class_weights[0])
    loss = focal_weight * bce * cw
    return loss.mean() if reduction == "mean" else loss


def compute_class_weights(labels, device=None) -> torch.Tensor:
    """Utility: compute balanced binary class weights from a list of 0/1 labels."""
    labels = torch.as_tensor(labels)
    num_neg = (labels == 0).sum().item()
    num_pos = (labels == 1).sum().item()
    n = num_neg + num_pos
    if num_neg == 0 or num_pos == 0:
        raise ValueError("Need at least one sample of each class to compute weights.")
    weights = torch.tensor([n / (2 * num_neg), n / (2 * num_pos)], dtype=torch.float32)
    if device is not None:
        weights = weights.to(device)
    return weights


def l1_norm_model(model: nn.Module) -> torch.Tensor:
    """L1 norm over trainable weights, skipping biases and normalization layers."""
    norm_classes = (
        nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm,
        nn.LayerNorm, nn.GroupNorm,
        nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
    )
    total = 0.0
    for module in model.modules():
        if isinstance(module, norm_classes):
            continue
        for param_name, p in module.named_parameters(recurse=False):
            if not p.requires_grad or param_name == "bias":
                continue
            total = total + p.abs().sum()
    return total
