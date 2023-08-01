# References:
    # https://gaussian37.github.io/vision-segmentation-miou/

import torch
import torch.nn as nn
import torch.nn.functional as F

IMG_SIZE = 513
N_CLASSES = 21


# "The performance is measured in terms of pixel intersection-over-union (IOU) averaged
# across the 21 classes."
class PixelmIoU(nn.Module):
    def forward(self, pred, gt):
        argmax = torch.argmax(pred, dim=1, keepdim=True)

        iou_sum = 0
        for c in range(N_CLASSES):
            pred_mask = (argmax == c)
            gt_mask = (gt == c)

            intersec = (pred_mask & gt_mask).sum().item()
            if intersec == 0:
                iou = 0
            else:
                union = (pred_mask | gt_mask).sum().item()
                
                iou = intersec / union

            iou_sum += iou
        miou = iou_sum / N_CLASSES
        return miou


if __name__ == "__main__":
    pred = torch.randn(16, N_CLASSES, IMG_SIZE, IMG_SIZE)
    metric = PixelmIoU()
    metric(pred=pred, gt=gt)


    # argmax = torch.argmax(pred, dim=1, keepdim=True)
    # # mask = (gt >= 0) & (gt < N_CLASSES)
    # # label = N_CLASSES * gt[mask] + argmax[mask]
    # label = N_CLASSES * gt + argmax
    # label
    # cnt = torch.bincount(label)
    # cnt.shape
    # cnt.reshape(N_CLASSES, N_CLASSES)


    # torch.bincount(gt.view(-1))[: N_CLASSES]
    # torch.bincount(argmax.view(-1))[: N_CLASSES]
    # # intersec = (argmax == gt)


    # input = torch.randint(0, 8, (5,), dtype=torch.int64)
    # weights = torch.linspace(0, 1, steps=5)
    # input, weights

    # torch.bincount(input)

    # input.bincount(weights)
