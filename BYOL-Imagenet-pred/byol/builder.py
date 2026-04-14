"""
BYOL + Augmentation Prediction model.

Architecture:
  Online network:
    - backbone: ResNet-50 (fc → Identity), outputs 2048-dim
    - projector: Linear(2048, 4096) → BN(4096) → ReLU → Linear(4096, 256)
    - predictor: Linear(256, 4096) → BN(4096) → ReLU → Linear(4096, 256)
    - aug_classifier: Linear(2048, num_pred_classes) on backbone features via avgpool hook

  Target network (momentum-updated copy, NO gradients):
    - backbone: copy of online backbone
    - projector: copy of online projector
    - NO predictor, NO aug_classifier

State dict keys: online_backbone.*, online_projector.*, online_predictor.*,
                 target_backbone.*, target_projector.*, aug_classifier.*
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_mlp(in_dim, hidden_dim, out_dim):
    """2-layer MLP with BN after the hidden layer (BYOL paper architecture)."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, out_dim),
    )


class BYOLPred(nn.Module):

    def __init__(self, base_encoder, proj_hidden_dim=4096, proj_dim=256,
                 pred_hidden_dim=4096, pred_dim=256, init_tau=0.996,
                 num_pred_classes=4):
        """
        Args:
            base_encoder: encoder factory (e.g. torchvision.models.resnet50)
            proj_hidden_dim: projector hidden dimension (default: 4096)
            proj_dim: projector output dimension (default: 256)
            pred_hidden_dim: predictor hidden dimension (default: 4096)
            pred_dim: predictor output dimension (default: 256)
            init_tau: initial momentum for target EMA (default: 0.996)
            num_pred_classes: number of augmentation combo classes (default: 4)
        """
        super().__init__()
        self.tau = init_tau

        # --- Online network ---
        encoder = base_encoder()
        feat_dim = encoder.fc.in_features  # 2048 for resnet50
        encoder.fc = nn.Identity()
        self.online_backbone = encoder

        self.online_projector = _build_mlp(feat_dim, proj_hidden_dim, proj_dim)
        self.online_predictor = _build_mlp(proj_dim, pred_hidden_dim, pred_dim)

        # --- Augmentation prediction (on online backbone features) ---
        self._backbone_feat = None
        self.aug_classifier = nn.Linear(feat_dim, num_pred_classes)

        # Hook on avgpool to capture pre-projection backbone features
        def _hook_fn(module, input, output):
            self._backbone_feat = output.flatten(1)  # (N, feat_dim)
        self.online_backbone.avgpool.register_forward_hook(_hook_fn)

        # --- Target network (no predictor, no aug_classifier) ---
        self.target_backbone = copy.deepcopy(self.online_backbone)
        self.target_projector = copy.deepcopy(self.online_projector)

        # Freeze target — no gradients ever
        for p in self.target_backbone.parameters():
            p.requires_grad = False
        for p in self.target_projector.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def _update_target(self):
        """Momentum update: ξ ← τ·ξ + (1-τ)·θ"""
        for online_p, target_p in zip(self.online_backbone.parameters(),
                                       self.target_backbone.parameters()):
            target_p.data = self.tau * target_p.data + (1 - self.tau) * online_p.data
        for online_p, target_p in zip(self.online_projector.parameters(),
                                       self.target_projector.parameters()):
            target_p.data = self.tau * target_p.data + (1 - self.tau) * online_p.data

    def forward(self, view1, view2):
        """
        Args:
            view1: (N, 3, 224, 224) first augmented view
            view2: (N, 3, 224, 224) second augmented view
        Returns:
            p1: (N, pred_dim) online prediction from view1, L2-normalized
            p2: (N, pred_dim) online prediction from view2, L2-normalized
            tz1: (N, proj_dim) target projection from view1, L2-normalized
            tz2: (N, proj_dim) target projection from view2, L2-normalized
            aug_logits1: (N, num_pred_classes) augmentation logits from view1
            aug_logits2: (N, num_pred_classes) augmentation logits from view2
        """
        # Online: view1
        h1 = self.online_backbone(view1)       # triggers hook
        aug_logits1 = self.aug_classifier(self._backbone_feat)
        z1 = self.online_projector(h1)
        p1 = self.online_predictor(z1)

        # Online: view2
        h2 = self.online_backbone(view2)       # triggers hook again
        aug_logits2 = self.aug_classifier(self._backbone_feat)
        z2 = self.online_projector(h2)
        p2 = self.online_predictor(z2)

        # Target (no gradients, no predictor, no aug_classifier)
        with torch.no_grad():
            th1 = self.target_backbone(view1)
            tz1 = self.target_projector(th1)
            th2 = self.target_backbone(view2)
            tz2 = self.target_projector(th2)

        # L2 normalize predictions and targets
        p1 = F.normalize(p1, dim=1)
        p2 = F.normalize(p2, dim=1)
        tz1 = F.normalize(tz1, dim=1)
        tz2 = F.normalize(tz2, dim=1)

        return p1, p2, tz1, tz2, aug_logits1, aug_logits2
