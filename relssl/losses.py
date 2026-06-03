"""
Losses for relssl.

  - NTXentLoss   : SimCLR contrastive loss (copied from SimCLR-Imagenet/simclr/loss.py)
  - BYOLLoss     : symmetrized BYOL regression loss (copied from BYOL-Imagenet/byol/loss.py)
  - RelPairLoss  : per-factor binary cross-entropy for the relational head, with
                   per-sample factor masking (saturation/hue under grayscale) handled
                   so a fully-masked factor in a batch is dropped, never divided-by-zero.

MoCo / LooC use plain nn.CrossEntropyLoss over InfoNCE logits, so no copy is needed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """NT-Xent loss. Input z: (2N, dim) L2-normalized; first N = view1, second N = view2."""

    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z):
        N2 = z.size(0)
        N = N2 // 2
        sim = torch.mm(z, z.t()) / self.temperature
        mask = torch.eye(N2, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, -1e9)
        pos_idx = torch.cat([
            torch.arange(N, N2, device=z.device),
            torch.arange(0, N, device=z.device),
        ])
        positives = sim[torch.arange(N2, device=z.device), pos_idx]
        loss = -positives + torch.logsumexp(sim, dim=1)
        return loss.mean()


class BYOLLoss(nn.Module):
    """Symmetrized regression loss: 2 - 2*<p, tz> on L2-normalized features."""

    def forward(self, p1, p2, tz1, tz2):
        loss1 = 2 - 2 * (p1 * tz2).sum(dim=1)
        loss2 = 2 - 2 * (p2 * tz1).sum(dim=1)
        return (loss1 + loss2).mean()


class RelPairLoss(nn.Module):
    """Per-factor BCE for the relational head.

    Args (forward):
        logits: (N, F) raw logits
        labels: (N, F) float in {0,1} (1 == "same")
        mask:   (N, F) float in {0,1} (0 == factor unobservable for that sample)

    Returns:
        loss: scalar = mean over factors (with >=1 active sample) of the per-factor
              mask-weighted BCE. Equal weight per factor regardless of mask frequency.
        acc_pct: (F,) per-factor accuracy in % over active samples (0 where inactive).
        active:  (F,) number of active samples per factor in this batch (for meter weights).
    """

    def forward(self, logits, labels, mask):
        bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")  # (N,F)
        active = mask.sum(dim=0)                                                     # (F,)
        denom = active.clamp(min=1.0)
        per_factor_loss = (bce * mask).sum(dim=0) / denom                            # (F,)
        valid = active > 0
        if valid.any():
            loss = per_factor_loss[valid].mean()
        else:  # pathological (all factors masked) — keep graph, contribute nothing
            loss = (logits.sum() * 0.0)

        with torch.no_grad():
            correct = ((logits > 0).float() == labels).float()                       # (N,F)
            acc_pct = (correct * mask).sum(dim=0) / denom * 100.0                     # (F,)
        return loss, acc_pct, active
