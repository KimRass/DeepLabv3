# References:
    # https://gaussian37.github.io/vision-segmentation-miou/
    # https://stackoverflow.com/questions/62461379/multiclass-semantic-segmentation-model-evaluation

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

import config
from model import DeepLabv3ResNet101
from voc2012 import VOC2012Dataset
from utils import modify_state_dict


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt_path", type=str, required=True)

    args = parser.parse_args()
    return args


# "The performance is measured in terms of pixel intersection-over-union (IOU) averaged
# across the 21 classes."
class PixelIoUByClass(nn.Module):
    def forward(self, pred, gt):
        argmax = torch.argmax(pred, dim=1, keepdim=True)

        ious = dict()
        for idx, c in enumerate(config.VOC_CLASSES):
            if c == "background":
                continue

            pred_mask = (argmax == idx)
            gt_mask = (gt == idx)
            if gt_mask.sum().item() == 0:
                continue

            union = (pred_mask | gt_mask).sum().item()
            intersec = (pred_mask & gt_mask).sum().item()
            iou = intersec / union

            ious[c] = round(iou, 4)
        return ious


@torch.no_grad()
def evaluate(val_dl, model, metric, device):
    model.eval()

    with torch.no_grad():
        sum_miou = 0
        for image, gt in tqdm(val_dl):
            image = image.to(device)
            gt = gt.to(device)
            pred = model(image)

            ious = metric(pred=pred, gt=gt)
            miou = sum(ious.values()) / len(ious)

            sum_miou += miou
    avg_miou = sum_miou / len(val_dl)
    print(f"Average mIoU: {avg_miou:.4f}")

    model.train()
    return avg_miou


if __name__ == "__main__":
    args = get_args()

    val_ds = VOC2012Dataset(img_dir=config.IMG_DIR, gt_dir=config.GT_DIR, split="val")
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=config.N_WORKERS)

    DEVICE = torch.device("cpu")
    model = DeepLabv3ResNet101(output_stride=16).to(DEVICE)
    state_dict = torch.load(args.ckpt_path, map_location=DEVICE)
    state_dict = modify_state_dict(state_dict["model"])
    model.load_state_dict(state_dict)

    metric = PixelIoUByClass()
    avg_miou = evaluate(val_dl=val_dl, model=model, metric=metric, device=DEVICE)
    print(f"Average mIoU: {avg_miou:.4f}")
