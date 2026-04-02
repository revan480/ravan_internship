"""
SimCLR model.

Architecture:
  - ResNet-50 backbone (fc replaced with Identity)
  - 2-layer MLP projection head: Linear(2048, 2048) -> ReLU -> Linear(2048, 128)
  - No momentum encoder, no queue

State dict keys: backbone.*, projector.*
"""

import torch.nn as nn
import torch.nn.functional as F


class SimCLR(nn.Module):

    def __init__(self, base_encoder, dim=128):
        """
        Args:
            base_encoder: encoder factory (e.g. torchvision.models.resnet50)
            dim: projection output dimension (default: 128)
        """
        super().__init__()

        # Backbone: standard ResNet with fc removed
        encoder = base_encoder()
        feat_dim = encoder.fc.in_features  # 2048 for resnet50
        encoder.fc = nn.Identity()
        self.backbone = encoder

        # 2-layer MLP projection head
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, dim),
        )

    def forward(self, x):
        """
        Args:
            x: (N, 3, 224, 224) input images
        Returns:
            z: (N, dim) L2-normalized projected features
        """
        h = self.backbone(x)       # (N, feat_dim)
        z = self.projector(h)       # (N, dim)
        z = F.normalize(z, dim=1)   # L2 normalize
        return z
