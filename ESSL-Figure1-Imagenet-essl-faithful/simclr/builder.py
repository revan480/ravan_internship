"""
SimCLR models for E-SSL Figure 1 experiments.

Two model classes:
  - SimCLR: standard contrastive model (invariance condition)
  - SimCLRPred: contrastive + augmentation prediction head (sensitivity condition)

Both use state dict prefix backbone.* for the ResNet encoder.
"""

import torch.nn as nn
import torch.nn.functional as F


class SimCLR(nn.Module):
    """Standard SimCLR model (no prediction head). Used in invariance condition."""

    def __init__(self, base_encoder, dim=128):
        super().__init__()

        encoder = base_encoder()
        feat_dim = encoder.fc.in_features  # 512 for resnet18, 2048 for resnet50
        encoder.fc = nn.Identity()
        self.backbone = encoder

        self.projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, dim),
        )

    def forward(self, x):
        h = self.backbone(x)
        z = self.projector(h)
        z = F.normalize(z, dim=1)
        return z


class SimCLRPred(nn.Module):
    """SimCLR + augmentation prediction head. Used in sensitivity condition."""

    def __init__(self, base_encoder, dim=128, num_pred_classes=4):
        super().__init__()

        encoder = base_encoder()
        feat_dim = encoder.fc.in_features
        encoder.fc = nn.Identity()
        self.backbone = encoder

        self.projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, dim),
        )

        self._backbone_feat = None
        self.aug_classifier = nn.Linear(feat_dim, num_pred_classes)

        def _hook_fn(module, input, output):
            self._backbone_feat = output.flatten(1)
        self.backbone.avgpool.register_forward_hook(_hook_fn)

    def forward(self, x):
        h = self.backbone(x)
        z = self.projector(h)
        z = F.normalize(z, dim=1)
        aug_logits = self.aug_classifier(self._backbone_feat)
        return z, aug_logits
