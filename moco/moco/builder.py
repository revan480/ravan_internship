"""
MoCo v2: Momentum Contrast with MLP projection head.

References:
  - He et al., "Momentum Contrast for Unsupervised Visual Representation Learning", CVPR 2020
  - Chen et al., "Improved Baselines with Momentum Contrastive Learning", arXiv 2020
"""

import torch
import torch.nn as nn


class MoCo(nn.Module):
    """
    MoCo v2 model: query encoder + momentum-updated key encoder + queue.
    """

    def __init__(self, base_encoder, dim=128, K=4096, m=0.999, T=0.2, mlp=True):
        """
        Args:
            base_encoder: encoder class (e.g. torchvision.models.resnet18)
            dim: feature dimension for contrastive loss (default: 128)
            K: queue size (default: 4096, smaller than ImageNet default since CUB is small)
            m: momentum for key encoder update (default: 0.999)
            T: temperature for softmax (default: 0.2)
            mlp: whether to use MLP projection head (True for MoCo v2)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # Create encoders
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        if mlp:
            # MoCo v2: replace fc with 2-layer MLP projection head
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp),
                nn.ReLU(),
                nn.Linear(dim_mlp, dim),
            )
            self.encoder_k.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp),
                nn.ReLU(),
                nn.Linear(dim_mlp, dim),
            )

        # Initialize key encoder with query encoder weights (no gradient)
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # Create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder: theta_k = m * theta_k + (1 - m) * theta_q"""
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        # If batch_size fits in the remaining space, simple copy
        if ptr + batch_size <= self.K:
            self.queue[:, ptr : ptr + batch_size] = keys.T
        else:
            # Wrap around
            remaining = self.K - ptr
            self.queue[:, ptr:] = keys.T[:, :remaining]
            self.queue[:, : batch_size - remaining] = keys.T[:, remaining:]

        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k):
        """
        Forward pass.

        Args:
            im_q: query images (N, C, H, W)
            im_k: key images (N, C, H, W)

        Returns:
            logits: (N, 1+K) contrastive logits
            labels: (N,) ground truth labels (all zeros)
        """
        # Compute query features
        q = self.encoder_q(im_q)  # (N, dim)
        q = nn.functional.normalize(q, dim=1)

        # Compute key features (no gradient)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k(im_k)  # (N, dim)
            k = nn.functional.normalize(k, dim=1)

        # Positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)

        # Negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])

        # Logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T

        # Labels: positives are the 0-th index
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # Dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels
