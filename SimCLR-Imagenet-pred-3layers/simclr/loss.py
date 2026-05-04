"""
NT-Xent (Normalized Temperature-scaled Cross Entropy) loss for SimCLR.

Given a batch of N images producing 2N augmented views, for each positive
pair (i, j) the 2(N-1) other views are negatives. Uses cosine similarity
between L2-normalized projections with temperature scaling.
"""

import torch
import torch.nn as nn


class NTXentLoss(nn.Module):

    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z):
        """
        Args:
            z: (2N, dim) tensor of L2-normalized projected features.
               First N are view1, second N are view2 of the same images.
        Returns:
            loss: scalar NT-Xent loss averaged over all 2N views.
        """
        N2 = z.size(0)
        N = N2 // 2

        # Cosine similarity matrix (2N x 2N)
        # z is already L2-normalized, so dot product = cosine similarity
        sim = torch.mm(z, z.t()) / self.temperature

        # Mask out self-similarity (diagonal) with large negative value
        mask = torch.eye(N2, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, -1e9)

        # Positive pair indices: view1[i] <-> view2[i]
        # For i in [0, N): positive is at index i + N
        # For i in [N, 2N): positive is at index i - N
        pos_idx = torch.cat([
            torch.arange(N, N2, device=z.device),
            torch.arange(0, N, device=z.device),
        ])

        # Gather positive similarities
        positives = sim[torch.arange(N2, device=z.device), pos_idx]

        # NT-Xent loss: -log(exp(pos) / sum_k!=i(exp(sim_ik)))
        # = -pos + logsumexp(sim, dim=1)
        loss = -positives + torch.logsumexp(sim, dim=1)

        return loss.mean()
