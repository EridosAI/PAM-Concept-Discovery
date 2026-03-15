"""PAM predictor architecture.

4-layer MLP with learned residual, GELU activation, and LayerNorm.
Transforms BGE embeddings into association-query space.

Architecture is intentionally identical to AAR v1 -- the variable
under test is data scale, not model capacity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AssociationMLP(nn.Module):
    """PAM predictor: transforms embeddings into association-query space.

    Default: 4-layer MLP, hidden 1024, ~4.2M params.
    Large variant (concept discovery): 8-layer, hidden 2048, ~29.4M params.
    """

    def __init__(self, embedding_dim: int = 1024, hidden_dim: int = 1024,
                 num_layers: int = 4):
        super().__init__()
        layers = []
        # Input projection
        layers.append(nn.Linear(embedding_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.GELU())
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
        # Output projection back to embedding dim
        layers.append(nn.Linear(hidden_dim, embedding_dim))
        self.net = nn.Sequential(*layers)
        # Learned residual -- association query is a modification, not replacement
        self.residual_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        transformed = self.net(x)
        alpha = torch.sigmoid(self.residual_weight)
        out = alpha * x + (1 - alpha) * transformed
        return F.normalize(out, dim=-1)
