# References:
    # https://gaussian37.github.io/vision-segmentation-miou/
    # https://stackoverflow.com/questions/62461379/multiclass-semantic-segmentation-model-evaluation

import torch
import torch.nn as nn

import config


# "The performance is measured in terms of pixel intersection-over-union (IOU) averaged
# across the 21 classes."
class PixelIoUByClass(nn.Module):
    def forward(self, pred, gt):
        argmax = torch.argmax(pred, dim=1, keepdim=True)

        ious = dict()
        for idx, c in enumerate(config.VOC_CLASSES):
            pred_mask = (argmax == idx)
            gt_mask = (gt == idx)
            if gt_mask.sum().item() == 0:
                continue

            union = (pred_mask | gt_mask).sum().item()
            intersec = (pred_mask & gt_mask).sum().item()
            iou = intersec / union

            ious[c] = round(iou, 4)
        # miou = sum(ious.values()) / len(ious)
        # return miou
        return ious


if __name__ == "__main__":
    metric = PixelIoUByClass()
    pred = torch.randn(16, config.N_CLASSES, config.IMG_SIZE, config.IMG_SIZE)
    metric(pred=pred, gt=gt)
