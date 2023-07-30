import torch.nn as nn

N_CLASSES = 21

class DeepLabLoss(nn.Module):
    def __init__(self):
        super().__init__()

        # "Our loss function is the sum of cross-entropy terms for each spatial position in the CNN output map
        # (subsampled by 8 compared to the original image). All positions and gts are equally weighted
        # in the overall loss function. Our targets are the ground truth gts (subsampled by 8)."
        self.ce = nn.CrossEntropyLoss(ignore_index=255, reduction="mean")

    def forward(self, pred, gt):
        pred = pred.permute(0, 2, 3, 1).reshape((-1, N_CLASSES))
        gt = gt.permute(0, 2, 3, 1).view(-1)
        loss = self.ce(pred, gt)
        return loss
