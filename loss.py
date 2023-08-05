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
        # self.ce = nn.CrossEntropyLoss(ignore_index=255, reduction="mean")
        self.ce = nn.CrossEntropyLoss(ignore_index=255, reduction="sum")

    def forward(self, pred, gt):
        pred = rearrange(pred, pattern="b c h w -> (b h w) c")
        gt = rearrange(gt, pattern="b c h w -> (b h w) c").squeeze(1)
        loss = self.ce(pred, gt)
        return loss


if __name__ == "__main__":
    ce = nn.CrossEntropyLoss(ignore_index=255, reduction="mean")
    image, gt = next(iter(train_dl))
    pred.shape, gt.shape
    pred = rearrange(pred, pattern="b c h w -> (b h w) c")
    gt = rearrange(gt, pattern="b c h w -> (b h w) c").squeeze(1)

    loss = ce(pred, gt)
