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
from torch.cuda.amp import GradScaler
import re
from collections import OrderedDict
import random
import os
import numpy as np

VOC_CLASS_COLOR = {
    "background": (0, 0, 0),
    "aeroplane": (128, 0, 0),
    "bicycle": (0, 128, 0),
    "bird": (128, 128, 0),
    "boat": (0, 0, 128),
    "bottle": (128, 0, 128),
    "bus": (0, 128, 128),
    "car": (128, 128, 128),
    "cat": (64, 0, 0),
    "chair": (192, 0, 0),
    "cow": (64, 128, 0),
    "diningtable": (192, 128, 0),
    "dog": (64, 0, 128),
    "horse": (192, 0, 128),
    "motorbike": (64, 128, 128),
    "person": (192, 128, 128),
    "pottedplant": (0, 64, 0),
    "sheep": (128, 64, 0),
    "sofa": (0, 192, 0),
    "train": (128, 192, 0),
    "tvmonitor": (0, 64, 128),
    "GRID": (255, 255, 255),
}
VOC_CLASSES = list(VOC_CLASS_COLOR.keys())[: -1]
N_CLASSES = len(VOC_CLASSES)
VOC_COLORS = list(VOC_CLASS_COLOR.values())


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    return device


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def get_grad_scaler(device):
    return GradScaler() if device.type == "cuda" else None


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
    grid.putpalette(sum(VOC_COLORS, ()))
    return grid.convert("RGB")


def visualize_batched_image_and_gt(image, gt, n_cols, alpha=0.7):
    image = visualize_batched_image(image, n_cols=n_cols)
    gt = visualize_batched_gt(gt, n_cols=n_cols)
    Image.blend(image, gt, alpha=alpha).show()


def modify_state_dict(state_dict, pattern=r"^module.|^_orig_mod."):
    new_state_dict = OrderedDict()
    for old_key, value in state_dict.items():
        new_key = re.sub(pattern=pattern, repl="", string=old_key)
        new_state_dict[new_key] = value
    return new_state_dict


def denorm(tensor, mean, std):
    return TF.normalize(
        tensor, mean=- np.array(mean) / np.array(std), std=1 / np.array(std),
    )


def image_to_grid(image, mean, std, n_cols, padding=1):
    tensor = image.clone().detach().cpu()
    tensor = denorm(tensor, mean=mean, std=std)
    grid = make_grid(tensor, nrow=n_cols, padding=1, pad_value=padding)
    grid.clamp_(0, 1)
    grid = TF.to_pil_image(grid)
    return grid
