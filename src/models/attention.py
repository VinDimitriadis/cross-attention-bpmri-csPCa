"""Cross-attention module across three sequences (T2W, DWI, ADC)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    """Three-stream cross-attention.

    Each sequence m in {i, j, k} produces Q_m, K_m, V_m through its own linear
    projection. The attended feature for stream m is obtained by attending its
    query against the keys/values of the *other* streams and projecting back
    through a shared output layer.

    Note on the asymmetric context selection:
            stream 0 (T2W) attends to: [stream 1]           (DWI)
            stream 1 (DWI) attends to: [stream 0]           (T2W)
            stream 2 (ADC) attends to: [stream 0, stream 1] (T2W + DWI)
        i.e. ADC serves as the integrating stream.
    """

    def __init__(self, feature_dim: int, dropout: float = 0.2):
        super().__init__()
        self.feature_dim = feature_dim
        self.scale = feature_dim ** -0.5

        self.query_layers = nn.ModuleList([nn.Linear(feature_dim, feature_dim) for _ in range(3)])
        self.key_layers = nn.ModuleList([nn.Linear(feature_dim, feature_dim) for _ in range(3)])
        self.value_layers = nn.ModuleList([nn.Linear(feature_dim, feature_dim) for _ in range(3)])
        self.out_projection = nn.Linear(feature_dim, feature_dim)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, f1: torch.Tensor, f2: torch.Tensor, f3: torch.Tensor):
        """Inputs are pooled sequence features of shape (B, feature_dim)."""
        features = [f1, f2, f3]

        qs = [self.query_layers[i](f).unsqueeze(1) for i, f in enumerate(features)]
        ks = [self.key_layers[i](f).unsqueeze(1) for i, f in enumerate(features)]
        vs = [self.value_layers[i](f).unsqueeze(1) for i, f in enumerate(features)]

        outputs = []
        for i in range(3):
            combined_ks = torch.cat([ks[j] for j in range(2) if j != i], dim=1)
            combined_vs = torch.cat([vs[j] for j in range(2) if j != i], dim=1)

            attn_scores = torch.matmul(qs[i], combined_ks.transpose(-2, -1)) * self.scale
            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_probs = self.dropout(attn_probs)

            attended = torch.matmul(attn_probs, combined_vs).squeeze(1)
            attended = self.dropout(attended)
            outputs.append(self.out_projection(attended))

        return outputs
