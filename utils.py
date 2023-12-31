# References:
    # https://github.com/fregu856/deeplabv3/blob/master/utils/utils.py

from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid
from time import time
from datetime import timedelta
import re
from collections import OrderedDict

import config

ROOT = Path(__file__).resolve().parent


def get_val_filenames(img_dir):
    val_txt_path = Path(img_dir).parent/"ImageSets/Segmentation/val.txt"
    with open(val_txt_path, mode="r") as f:
        filenames = [l.strip() for l in f.readlines()]
    return filenames


def get_voc2012_trainaug_mean_and_std(img_dir, gt_dir):
    img_dir = Path(img_dir)
    gt_dir = Path(gt_dir)

    val_filenames = get_val_filenames(img_dir)
    cnt = 0
    sum_rgb = 0
    sum_rgb_square = 0
    sum_resol = 0
    for gt_path in tqdm(list(Path(gt_dir).glob("*.png"))):
        if gt_path.stem in val_filenames:
            continue
        cnt += 1

        img_path = (img_dir/gt_path.stem).with_suffix(".jpg")
        pil_img = Image.open(img_path)
        tensor = T.ToTensor()(pil_img)
        
        sum_rgb += tensor.sum(dim=(1, 2))
        sum_rgb_square += (tensor ** 2).sum(dim=(1, 2))
        _, h, w = tensor.shape
        sum_resol += h * w
    mean = torch.round(sum_rgb / sum_resol, decimals=3)
    std = torch.round((sum_rgb_square / sum_resol - mean ** 2) ** 0.5, decimals=3)
    print(f"""Total {cnt:,} images found.""")
    return mean, std


def get_image_dataset_mean_and_std(data_dir, ext="jpg"):
    data_dir = Path(data_dir)

    sum_rgb = 0
    sum_rgb_square = 0
    sum_resol = 0
    for img_path in tqdm(list(data_dir.glob(f"""**/*.{ext}"""))):
        pil_img = Image.open(img_path)
        tensor = T.ToTensor()(pil_img)
        
        sum_rgb += tensor.sum(dim=(1, 2))
        sum_rgb_square += (tensor ** 2).sum(dim=(1, 2))
        _, h, w = tensor.shape
        sum_resol += h * w
    mean = torch.round(sum_rgb / sum_resol, decimals=3)
    std = torch.round((sum_rgb_square / sum_resol - mean ** 2) ** 0.5, decimals=3)
    return mean, std


def get_elapsed_time(start_time):
    return timedelta(seconds=round(time() - start_time))


def visualize_batched_image(image, n_cols):
    grid = make_grid(image, nrow=n_cols, normalize=True, pad_value=1)
    grid = TF.to_pil_image(grid)
    return grid


def visualize_batched_gt(gt, n_cols):
    """
    Args:
        gt: `(b, 1, h, w)` (dtype: `torch.long()`)
    """
    gt[gt == 255] = 0
    grid = make_grid(gt, nrow=n_cols, pad_value=21)
    grid = Image.fromarray(grid[0].numpy().astype("uint8"), mode="P")
    grid.putpalette(sum(config.VOC_COLORS, ()))
    return grid.convert("RGB")


def visualize_batched_image_and_gt(image, gt, n_cols, alpha=0.7):
    image = visualize_batched_image(image, n_cols=n_cols)
    gt = visualize_batched_gt(gt, n_cols=n_cols)
    return Image.blend(image, gt, alpha=alpha)


def modify_state_dict(state_dict, pattern=r"^module.|^_orig_mod."):
    new_state_dict = OrderedDict()
    for old_key, value in state_dict.items():
        new_key = re.sub(pattern=pattern, repl="", string=old_key)
        new_state_dict[new_key] = value
    return new_state_dict
