import torch
import torch.nn as nn


class LooC(nn.Module):
    """
    LooC model: shared backbone + multiple projection heads + per-head queues.

    Architecture:
        backbone_q / backbone_k: ResNet-50 up to avgpool (2048-d)
        heads_q / heads_k: n_aug+1 MLP projection heads (2048 → 2048 → 128)
        queue_i: one queue per embedding space, shape [dim, K]

    Embedding spaces (n_aug+1 total):
        Z0: all-invariant (standard MoCo objective)
        Z1..Zn: one per atomic augmentation (variant in that augmentation)
    """

    def __init__(self, base_encoder, dim=128, K=16384, m=0.999, T=0.2, mlp=True, n_aug=2, lambda_pred=0.0):
        """
        Args:
            base_encoder: encoder class (e.g. torchvision.models.resnet50)
            dim: projection dimension for contrastive loss (default: 128)
            K: queue size per embedding space (default: 16384)
            m: momentum for key encoder update (default: 0.999)
            T: temperature for softmax (default: 0.2)
            mlp: unused, kept for API compatibility (LooC always uses MLP heads)
            n_aug: number of atomic augmentations (default: 2 for rotation + color)
            lambda_pred: weight for augmentation prediction loss (0.0 = disabled)
        """
        super(LooC, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.n_aug = n_aug
        self.n_heads = n_aug + 1  # Z0 + one per augmentation
        self.lambda_pred = lambda_pred

        # Create backbone encoders (ResNet-50 with fc replaced by Identity)
        self.backbone_q = base_encoder()
        self.backbone_k = base_encoder()
        feat_dim = self.backbone_q.fc.in_features  # 2048 for ResNet-50
        self.backbone_q.fc = nn.Identity()
        self.backbone_k.fc = nn.Identity()
        #Moco -> encoder is one piece - backbone + projection head fused together
        #Looc -> separate because multiple heads share one backbone

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

        # Initialize key encoder with query encoder weights
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
        for i in range(self.n_heads): #3 quees: q_0, q_1, q_2
            self.register_buffer(f"queue_{i}", torch.randn(dim, K))
            setattr(self, f"queue_{i}",
                    nn.functional.normalize(getattr(self, f"queue_{i}"), dim=0))
            self.register_buffer(f"queue_ptr_{i}", torch.zeros(1, dtype=torch.long))

        self.criterion = nn.CrossEntropyLoss()

        # Augmentation prediction classifier (rotation angle: 3 classes)
        self.aug_classifier = nn.Linear(feat_dim, 3)
        self.aug_criterion = nn.CrossEntropyLoss()

    # Paper says: We adopt MOCO as the backbone of our framework
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

    def forward(self, views, aug_labels=None):
        """
        Forward pass for LooC.

        Args:
            views: list of n_aug+2 tensors [q, k0, k1, ...], each (N, C, H, W)
                q:   all augmentations sampled independently (query)
                k0:  all augmentations sampled independently (standard key, positive in Z0)
                k_i: shares augmentation i with q (positive in Z_{i+1})

        Returns:
            loss: scalar, average over all n_aug+1 embedding spaces
            logits0: (N, 1+K) contrastive logits in Z0 (for accuracy logging)
            labels0: (N,) all zeros (for accuracy logging)
        """
        im_q = views[0]
        im_k0 = views[1]
        im_keys = views[2:]  # n_aug extra key views

        # ---- Query features (with gradient) ----
        feat_q = self.backbone_q(im_q)  # (N, 2048)

        # ---- Key features and projections (no gradient, momentum encoder) ----
        with torch.no_grad():
            self._momentum_update_key_encoder()

            feat_k0 = self.backbone_k(im_k0)  # (N, 2048)
            feats_k = [self.backbone_k(k) for k in im_keys]

            # Project k0 through all heads
            k0_proj = [nn.functional.normalize(self.heads_k[h](feat_k0), dim=1)
                       for h in range(self.n_heads)]

            # Project each extra key through all heads
            # keys_proj[i][h] = extra key i projected through head h
            keys_proj = []
            for feat_ki in feats_k:
                keys_proj.append([
                    nn.functional.normalize(self.heads_k[h](feat_ki), dim=1)
                    for h in range(self.n_heads)
                ])

        # ---- Query projections (with gradient) ----
        q_proj = [nn.functional.normalize(self.heads_q[h](feat_q), dim=1)
                  for h in range(self.n_heads)]

        # ---- L0: standard MoCo InfoNCE in Z0 ----
        l0_pos = torch.einsum("nc,nc->n", [q_proj[0], k0_proj[0]]).unsqueeze(-1)  # (N, 1)
        l0_neg = torch.einsum("nc,ck->nk", [q_proj[0], self.queue_0.clone().detach()])  # (N, K)
        logits0 = torch.cat([l0_pos, l0_neg], dim=1) / self.T  # (N, 1+K)
        labels0 = torch.zeros(logits0.shape[0], dtype=torch.long, device=im_q.device)
        total_loss = self.criterion(logits0, labels0)

        # ---- Li: augmentation-variant InfoNCE in Zi (i = 1..n_aug) ----
        for i in range(self.n_aug):
            head_idx = i + 1
            qi = q_proj[head_idx]

            # Positive: extra key i shares augmentation i with q
            li_pos = torch.einsum("nc,nc->n", [qi, keys_proj[i][head_idx]]).unsqueeze(-1)

            # Same-image negatives: k0, plus all other extra keys (they don't share aug i)
            neg_parts = [torch.einsum("nc,nc->n", [qi, k0_proj[head_idx]]).unsqueeze(-1)]
            for j in range(self.n_aug):
                if j != i:
                    neg_parts.append(
                        torch.einsum("nc,nc->n", [qi, keys_proj[j][head_idx]]).unsqueeze(-1)
                    )

            # Queue negatives
            queue_i = getattr(self, f"queue_{head_idx}").clone().detach()
            neg_parts.append(torch.einsum("nc,ck->nk", [qi, queue_i]))

            logits_i = torch.cat([li_pos] + neg_parts, dim=1) / self.T
            labels_i = torch.zeros(logits_i.shape[0], dtype=torch.long, device=im_q.device)
            total_loss = total_loss + self.criterion(logits_i, labels_i)

        # ---- Enqueue k0 projections into all queues ----
        for h in range(self.n_heads):
            self._dequeue_and_enqueue(k0_proj[h], h)

        # ---- Augmentation prediction loss (if enabled) ----
        if self.lambda_pred > 0 and aug_labels is not None:
            aug_logits = self.aug_classifier(feat_q)
            loss_pred = self.aug_criterion(aug_logits, aug_labels)
            loss = total_loss / self.n_heads + self.lambda_pred * loss_pred
            aug_acc = (aug_logits.argmax(dim=1) == aug_labels).float().mean().item() * 100.0
            return loss, logits0, labels0, loss_pred.item(), aug_acc
        else:
            loss = total_loss / self.n_heads
            return loss, logits0, labels0, 0.0, 0.0
