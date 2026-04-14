"""
BYOL loss: symmetrized MSE between L2-normalized predictions and targets.

Unlike contrastive methods (MoCo, SimCLR), BYOL uses NO negative samples.
The loss is purely a regression objective — predict the target network's
representation of one view from the other view's online prediction.
"""

import torch.nn as nn


class BYOLLoss(nn.Module):

    def forward(self, p1, p2, tz1, tz2):
        """
        Symmetrized regression loss.

        Args:
            p1: (N, dim) online prediction from view1, L2-normalized
            p2: (N, dim) online prediction from view2, L2-normalized
            tz1: (N, dim) target projection from view1, L2-normalized, detached
            tz2: (N, dim) target projection from view2, L2-normalized, detached
        Returns:
            scalar loss
        """
        # ||p - tz||² = 2 - 2*<p, tz> when both are L2-normalized
        loss1 = 2 - 2 * (p1 * tz2).sum(dim=1)  # predict target-view2 from view1
        loss2 = 2 - 2 * (p2 * tz1).sum(dim=1)  # predict target-view1 from view2
        return (loss1 + loss2).mean()
