# References:
    # https://github.com/VainF/DeepLabV3Plus-Pytorch/blob/master/utils/loss.py

import torch.nn as nn
from einops import rearrange


class DeepLabLoss(nn.Module):
    def __init__(self):
        super().__init__()

        # "Our loss function is the sum of cross-entropy terms for each spatial position in the CNN output map.
        # All positions and labels are equally weighted in the overall loss function.
        # Our targets are the ground truth labels."
        self.ce = nn.CrossEntropyLoss(ignore_index=255, reduction="mean")

    def forward(self, pred, gt):
        pred = rearrange(pred, pattern="b c h w -> (b h w) c")
        gt = rearrange(gt, pattern="b c h w -> (b h w) c").squeeze(1)
        loss = self.ce(pred, gt)
        return loss


if __name__ == "__main__":
    crit = DeepLabLoss()
    # gt = gt[None, ...]
    pred = torch.randn(16, 21, 513, 513)
    pred.shape, gt.shape
    crit(pred=pred, gt=gt)
