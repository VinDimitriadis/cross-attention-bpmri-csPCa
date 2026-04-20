"""Full multimodal model: three VGG encoders + cross-attention + clinical variables."""
import torch
import torch.nn as nn

from .attention import CrossAttention
from .encoder import VGGEncoder
from .heads import ClassificationHead
from .layers import GlobalMaxAvgPool


class MultimodalProstateModel(nn.Module):
    """Per-sequence 3D VGG encoders → cross-attention → concat with clinical vars → classifier.

    Args:
        in_channel: Number of input channels per sequence (1 for greyscale MRI).
        label_category_dict: e.g. ``{'binary_task': 1}`` for binary classification.
        feature_dim: Encoder final stage width (fed into CrossAttention).
        num_clinical: Number of clinical scalar variables concatenated at the end.
        dim: Spatial dimensionality (3 for volumetric data).
    """

    def __init__(self,
                 in_channel: int,
                 label_category_dict: dict,
                 feature_dim: int = 512,
                 num_clinical: int = 3,
                 attention_dropout: float = 0.2,
                 dim: int = 3):
        super().__init__()

        encoder_kwargs = dict(
            stage_output_channels=(64, 128, 256, 512),
            blocks=(1, 2, 3, 4),
            downsample_ratio=(0.5, 0.5, 0.5, 0.5),
            norm="bn",
            active="relu",
            dim=dim,
        )

        self.encoder_t2w = VGGEncoder(in_channel, **encoder_kwargs)
        self.encoder_dwi = VGGEncoder(in_channel, **encoder_kwargs)
        self.encoder_adc = VGGEncoder(in_channel, **encoder_kwargs)

        self.cross_attention = CrossAttention(feature_dim=feature_dim,
                                              dropout=attention_dropout)
        self.pooling = GlobalMaxAvgPool()

        # 3 attended sequence features + clinical variables
        self.cls_head = ClassificationHead(
            label_category_dict,
            in_channel=feature_dim * 3 + num_clinical,
        )

    def forward(self,
                t2w: torch.Tensor,
                dwi: torch.Tensor,
                adc: torch.Tensor,
                clinical: torch.Tensor) -> dict:
        # Encode each sequence and pool the last stage feature map
        f_t2w = self.pooling(self.encoder_t2w(t2w)[-1])
        f_dwi = self.pooling(self.encoder_dwi(dwi)[-1])
        f_adc = self.pooling(self.encoder_adc(adc)[-1])

        # Cross-attention across sequences
        a_t2w, a_dwi, a_adc = self.cross_attention(f_t2w, f_dwi, f_adc)

        # Concatenate attended features + clinical variables, then classify
        fused = torch.cat([a_t2w, a_dwi, a_adc, clinical], dim=1)
        return self.cls_head(fused)
