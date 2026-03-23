"""
MoCo v2: Momentum Contrast with MLP projection head.
Optionally includes an augmentation combination prediction classifier.

References:
  - He et al., "Momentum Contrast for Unsupervised Visual Representation Learning", CVPR 2020
  - Chen et al., "Improved Baselines with Momentum Contrastive Learning", arXiv 2020
"""

import torch
import torch.nn as nn


class MoCo(nn.Module):
    """
    MoCo v2 model: query encoder + momentum-updated key encoder + queue.
    Optionally adds an augmentation prediction classifier on backbone features.
    """

    def __init__(self, base_encoder, dim=128, K=16384, m=0.999, T=0.2, mlp=True,
                 lambda_pred=0.0, num_aug_classes=4):
        """
        Args:
            base_encoder: encoder class (e.g. torchvision.models.resnet50)
            dim: feature dimension for contrastive loss (default: 128)
            K: queue size (default: 16384 for ImageNet-100)
            m: momentum for key encoder update (default: 0.999)
            T: temperature for softmax (default: 0.2)
            mlp: whether to use MLP projection head (True for MoCo v2)
            lambda_pred: weight for augmentation prediction loss (0=disabled)
            num_aug_classes: number of augmentation combination classes (default: 4)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.lambda_pred = lambda_pred

        # Create encoders
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        # Capture backbone feature dimension before replacing fc
        # (2048 for ResNet-50, 512 for ResNet-18)
        dim_mlp = self.encoder_q.fc.weight.shape[1]

        if mlp:
            # MoCo v2: replace fc with 2-layer MLP projection head
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

        # Augmentation combination prediction classifier
        self._backbone_feat = None
        if self.lambda_pred > 0:
            # Classifier takes backbone features BEFORE the MLP projection head.
            # This is the 2048-dim output of avgpool (for ResNet-50).
            self.aug_classifier = nn.Linear(dim_mlp, num_aug_classes)

            # Register a forward hook on encoder_q's avgpool to capture
            # pre-projection features. The hook only fires for encoder_q,
            # NOT encoder_k (they are separate nn.Module instances), so
            # self._backbone_feat is always from the query encoder.
            def _hook_fn(module, input, output):
                self._backbone_feat = output.flatten(1)  # (N, dim_mlp)
            self.encoder_q.avgpool.register_forward_hook(_hook_fn)

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

    def forward(self, im_q, im_k, aug_labels=None):
        """
        Forward pass.

        Args:
            im_q: query images (N, C, H, W)
            im_k: key images (N, C, H, W)
            aug_labels: augmentation combination labels (N,), optional

        Returns:
            logits: (N, 1+K) contrastive logits
            labels: (N,) ground truth labels (all zeros)
            pred_loss: scalar augmentation prediction loss (0 if disabled)
            aug_acc: augmentation prediction accuracy in % (0 if disabled)
        """
        # Compute query features.
        # NOTE: self._backbone_feat is populated by the avgpool hook during this call.
        # It must be read AFTER this line and BEFORE any other encoder_q forward call.
        # encoder_k is a separate module, so its forward does NOT trigger this hook.
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

        # Augmentation prediction branch
        if self.lambda_pred > 0 and aug_labels is not None:
            # self._backbone_feat was captured by the hook during encoder_q(im_q) above.
            # Gradients flow from pred_loss → aug_classifier → backbone (through the hook).
            aug_logits = self.aug_classifier(self._backbone_feat)
            pred_loss = nn.CrossEntropyLoss()(aug_logits, aug_labels)
            aug_acc = (aug_logits.argmax(dim=1) == aug_labels).float().mean().item() * 100.0
            return logits, labels, pred_loss, aug_acc

        return logits, labels, torch.tensor(0.0, device=im_q.device), 0.0
