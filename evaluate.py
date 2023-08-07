# References:
    # https://gaussian37.github.io/vision-segmentation-miou/

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


# "The performance is measured in terms of pixel intersection-over-union (IOU) averaged
# across the 21 classes."
class PixelmIoU(nn.Module):
    def forward(self, pred, gt):
        argmax = torch.argmax(pred, dim=1, keepdim=True)

        ious = list()
        for c in range(config.N_CLASSES):
            pred_mask = (argmax == c)
            gt_mask = (gt == c)
            if gt_mask.sum().item() == 0:
                iou = None

            union = (pred_mask | gt_mask).sum().item()
            intersec = (pred_mask & gt_mask).sum().item()
            iou = intersec / union

            ious.append(iou)
        # miou = sum(ious) / len(ious)
        miou = np.nanmean(ious)
        return miou


if __name__ == "__main__":
    metric = PixelmIoU()
    pred = torch.randn(16, config.N_CLASSES, config.IMG_SIZE, config.IMG_SIZE)
    metric(pred=pred, gt=gt)
