import torch
import torch.nn as nn


class LooC(nn.Module):
    """
    LooC model: shared backbone + multiple projection heads + per-head queues.

    Architecture:
        backbone_q / backbone_k: ResNet-50 up to avgpool (2048-d)
        heads_q / heads_k: n_aug+1 MLP projection heads (2048 → 2048 → 128)
        queue_i: one queue per embedding space, shape [dim, K]

    Embedding spaces (for n_aug=2, rotation + color):
        Z0: all-invariant (standard MoCo objective)
        Z1: rotation-variant (sensitive to rotation, invariant to color)
        Z2: color-variant (sensitive to color, invariant to rotation)
    """

    def __init__(self, base_encoder, dim=128, K=16384, m=0.999, T=0.2, mlp=True, n_aug=2):
        """
        Args:
            base_encoder: encoder class (e.g. torchvision.models.resnet50)
            dim: projection dimension for contrastive loss (default: 128)
            K: queue size per embedding space (default: 16384)
            m: momentum for key encoder update (default: 0.999)
            T: temperature for softmax (default: 0.2)
            mlp: unused, kept for API compatibility (LooC always uses MLP heads)
            n_aug: number of atomic augmentations (default: 2 for rotation + color)
        """
        super(LooC, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.n_aug = n_aug
        self.n_heads = n_aug + 1  # Z0 + one per augmentation

        # Create backbone encoders (ResNet-50 with fc replaced by Identity)
        self.backbone_q = base_encoder()
        self.backbone_k = base_encoder()
        feat_dim = self.backbone_q.fc.in_features  # 2048 for ResNet-50
        self.backbone_q.fc = nn.Identity()
        self.backbone_k.fc = nn.Identity()

        # Create projection heads: n_aug+1 MLP heads
        # Each: Linear(2048, 2048) → ReLU → Linear(2048, 128)
        self.heads_q = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.ReLU(),
                nn.Linear(feat_dim, dim),
            )
            for _ in range(self.n_heads)
        ])
        self.heads_k = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.ReLU(),
                nn.Linear(feat_dim, dim),
            )
            for _ in range(self.n_heads)
        ])

        # Initialize key encoder with query encoder weights (no gradient)
        for param_q, param_k in zip(
            self.backbone_q.parameters(), self.backbone_k.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        for param_q, param_k in zip(
            self.heads_q.parameters(), self.heads_k.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # Create per-head queues
        for i in range(self.n_heads):
            self.register_buffer(f"queue_{i}", torch.randn(dim, K))
            setattr(self, f"queue_{i}",
                    nn.functional.normalize(getattr(self, f"queue_{i}"), dim=0))
            self.register_buffer(f"queue_ptr_{i}", torch.zeros(1, dtype=torch.long))

        self.criterion = nn.CrossEntropyLoss()

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update: theta_k = m * theta_k + (1 - m) * theta_q"""
        for param_q, param_k in zip(
            self.backbone_q.parameters(), self.backbone_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

        for param_q, param_k in zip(
            self.heads_q.parameters(), self.heads_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, queue_idx):
        """Enqueue keys into the specified queue (FIFO circular buffer)."""
        queue = getattr(self, f"queue_{queue_idx}")
        queue_ptr = getattr(self, f"queue_ptr_{queue_idx}")

        batch_size = keys.shape[0]
        ptr = int(queue_ptr)

        if ptr + batch_size <= self.K:
            queue[:, ptr : ptr + batch_size] = keys.T
        else:
            remaining = self.K - ptr
            queue[:, ptr:] = keys.T[:, :remaining]
            queue[:, : batch_size - remaining] = keys.T[:, remaining:]

        ptr = (ptr + batch_size) % self.K
        queue_ptr[0] = ptr

    def forward(self, views):
        """
        Forward pass for LooC.

        Args:
            views: list of 4 tensors [q, k0, k1, k2], each (N, C, H, W)
                q:  all augmentations sampled independently (query)
                k0: all augmentations sampled independently (standard key, positive in Z0)
                k1: shares rotation with q, fresh color (positive in Z1)
                k2: shares color with q, fresh rotation (positive in Z2)

        Returns:
            loss: scalar, average of L0 + L1 + L2
            logits0: (N, 1+K) contrastive logits in Z0 (for accuracy logging)
            labels0: (N,) all zeros (for accuracy logging)
        """
        im_q, im_k0, im_k1, im_k2 = views

        # ---- Query features (with gradient) ----
        feat_q = self.backbone_q(im_q)  # (N, 2048)

        # ---- Key features and projections (no gradient, momentum encoder) ----
        with torch.no_grad():
            self._momentum_update_key_encoder()
            feat_k0 = self.backbone_k(im_k0)  # (N, 2048)
            feat_k1 = self.backbone_k(im_k1)  # (N, 2048)
            feat_k2 = self.backbone_k(im_k2)  # (N, 2048)

            # Key projections (Z0)
            k0_z0 = nn.functional.normalize(self.heads_k[0](feat_k0), dim=1)

            # Key projections (Z1)
            k1_z1 = nn.functional.normalize(self.heads_k[1](feat_k1), dim=1)  # positive (same rotation as q)
            k0_z1 = nn.functional.normalize(self.heads_k[1](feat_k0), dim=1)  # negative (different rotation)
            k2_z1 = nn.functional.normalize(self.heads_k[1](feat_k2), dim=1)  # negative (different rotation)

            # Key projections (Z2)
            k2_z2 = nn.functional.normalize(self.heads_k[2](feat_k2), dim=1)  # positive (same color as q)
            k0_z2 = nn.functional.normalize(self.heads_k[2](feat_k0), dim=1)  # negative (different color)
            k1_z2 = nn.functional.normalize(self.heads_k[2](feat_k1), dim=1)  # negative (different color)

        # ---- Query projections (with gradient) ----
        q0 = nn.functional.normalize(self.heads_q[0](feat_q), dim=1)
        q1 = nn.functional.normalize(self.heads_q[1](feat_q), dim=1)
        q2 = nn.functional.normalize(self.heads_q[2](feat_q), dim=1)

        # ---- L0: standard MoCo InfoNCE in Z0 ----
        l0_pos = torch.einsum("nc,nc->n", [q0, k0_z0]).unsqueeze(-1)  # (N, 1)
        l0_neg = torch.einsum("nc,ck->nk", [q0, self.queue_0.clone().detach()])  # (N, K)
        logits0 = torch.cat([l0_pos, l0_neg], dim=1) / self.T  # (N, 1+K)
        labels0 = torch.zeros(logits0.shape[0], dtype=torch.long, device=im_q.device)
        loss0 = self.criterion(logits0, labels0)

        # ---- L1: rotation-variant InfoNCE in Z1 ----
        # Positive: k1 (shares rotation with q)
        # Same-image negatives: k0, k2 (different rotation from q)
        # Queue negatives: queue_1
        l1_pos = torch.einsum("nc,nc->n", [q1, k1_z1]).unsqueeze(-1)  # (N, 1)
        l1_neg_k0 = torch.einsum("nc,nc->n", [q1, k0_z1]).unsqueeze(-1)  # (N, 1)
        l1_neg_k2 = torch.einsum("nc,nc->n", [q1, k2_z1]).unsqueeze(-1)  # (N, 1)
        l1_neg_queue = torch.einsum("nc,ck->nk", [q1, self.queue_1.clone().detach()])  # (N, K)
        logits1 = torch.cat([l1_pos, l1_neg_k0, l1_neg_k2, l1_neg_queue], dim=1) / self.T  # (N, 1+2+K)
        labels1 = torch.zeros(logits1.shape[0], dtype=torch.long, device=im_q.device)
        loss1 = self.criterion(logits1, labels1)

        # ---- L2: color-variant InfoNCE in Z2 ----
        # Positive: k2 (shares color with q)
        # Same-image negatives: k0, k1 (different color from q)
        # Queue negatives: queue_2
        l2_pos = torch.einsum("nc,nc->n", [q2, k2_z2]).unsqueeze(-1)  # (N, 1)
        l2_neg_k0 = torch.einsum("nc,nc->n", [q2, k0_z2]).unsqueeze(-1)  # (N, 1)
        l2_neg_k1 = torch.einsum("nc,nc->n", [q2, k1_z2]).unsqueeze(-1)  # (N, 1)
        l2_neg_queue = torch.einsum("nc,ck->nk", [q2, self.queue_2.clone().detach()])  # (N, K)
        logits2 = torch.cat([l2_pos, l2_neg_k0, l2_neg_k1, l2_neg_queue], dim=1) / self.T  # (N, 1+2+K)
        labels2 = torch.zeros(logits2.shape[0], dtype=torch.long, device=im_q.device)
        loss2 = self.criterion(logits2, labels2)

        # ---- Enqueue k0 into all queues (reuse already-computed projections) ----
        self._dequeue_and_enqueue(k0_z0, 0)
        self._dequeue_and_enqueue(k0_z1, 1)
        self._dequeue_and_enqueue(k0_z2, 2)

        # ---- Average loss across all 3 embedding spaces ----
        loss = (loss0 + loss1 + loss2) / 3.0

        return loss, logits0, labels0
