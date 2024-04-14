import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import argparse

import config
from voc2012 import VOC2012Dataset
from model import DeepLabv3ResNet101
from utils import ROOT, modify_state_dict, visualize_batched_image_and_gt


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt_path", type=str, required=True)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    val_ds = VOC2012Dataset(img_dir=config.IMG_DIR, gt_dir=config.GT_DIR, split="val")
    val_dl = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=config.N_WORKERS)

    DEVICE = torch.device("cpu")
    model = DeepLabv3ResNet101(output_stride=16).to(DEVICE)
    state_dict = torch.load(args.ckpt_path, map_location=DEVICE)
    state_dict = modify_state_dict(state_dict["model"])
    model.load_state_dict(state_dict)

    with torch.no_grad():
        for batch, (image, gt) in enumerate(tqdm(val_dl), start=1):
            image = image.to(DEVICE)
            gt = gt.to(DEVICE)

            pred = model(image)
            argmax = torch.argmax(pred, dim=1, keepdim=True)

            gt_vis = visualize_batched_image_and_gt(image=image, gt=gt, n_cols=4, alpha=0.7)
            gt_vis.save(ROOT/f"voc2012_val_predictions/{batch}_gt.jpg")

            pred_vis = visualize_batched_image_and_gt(image=image, gt=argmax, n_cols=4, alpha=0.7)
            pred_vis.save(ROOT/f"voc2012_val_predictions/{batch}_pred.jpg")
