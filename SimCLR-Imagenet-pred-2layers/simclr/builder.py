"""
SimCLR + Augmentation Prediction model.

Architecture:
  - ResNet-50 backbone (fc replaced with Identity)
  - 2-layer MLP projection head: Linear(2048, 2048) -> ReLU -> Linear(2048, 128)
  - Augmentation combo classifier on backbone features (via avgpool hook)
  - No momentum encoder, no queue

State dict keys: backbone.*, projector.*, aug_classifier.*
"""

import torch.nn as nn
import torch.nn.functional as F


class SimCLRPred(nn.Module):

    def __init__(self, base_encoder, dim=128, num_pred_classes=4):
        """
        Args:
            base_encoder: encoder factory (e.g. torchvision.models.resnet50)
            dim: projection output dimension (default: 128)
            num_pred_classes: number of augmentation combo classes (default: 4)
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

        # Augmentation prediction classifier on backbone features
        self._backbone_feat = None
        self.aug_classifier = nn.Sequential(
            nn.Linear(feat_dim, 2048),
            nn.LayerNorm(2048),
            nn.ReLU(),
            nn.Linear(2048, num_pred_classes),
        )

        # Hook on avgpool to capture pre-projection backbone features
        def _hook_fn(module, input, output):
            self._backbone_feat = output.flatten(1)  # (N, feat_dim)
        self.backbone.avgpool.register_forward_hook(_hook_fn)

    def forward(self, x):
        """
        Args:
            x: (N, 3, 224, 224) input images
        Returns:
            z: (N, dim) L2-normalized projected features
            aug_logits: (N, num_pred_classes) augmentation prediction logits
        """
        h = self.backbone(x)                        # (N, feat_dim) — triggers hook
        z = self.projector(h)                        # (N, dim)
        z = F.normalize(z, dim=1)                    # L2 normalize
        aug_logits = self.aug_classifier(self._backbone_feat)  # (N, num_pred_classes)
        return z, aug_logits
