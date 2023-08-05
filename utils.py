# References:
    # https://github.com/fregu856/deeplabv3/blob/master/utils/utils.py

from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
import torch
import torchvision.transforms as T
from time import time
from datetime import timedelta

IMG_SIZE = 513

VOC_COLORMAP = [
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128],
]

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


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"""Using {torch.cuda.device_count()} GPU(s).""")
    else:
        device = torch.device("cpu")
        print("Using CPU.")
    return device


def get_elapsed_time(start_time):
    return timedelta(seconds=round(time() - start_time))


def save_checkpoint(step, n_steps, model, optim, save_path):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "step": step,
        "number_of_steps": n_steps,
        "model": model.state_dict(),
        "optimizer": optim.state_dict(),
    }
    torch.save(ckpt, str(save_path))


import numpy as np
from skimage.io import imshow
import matplotlib.pyplot as plt

def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap


def label_img_to_color(img):
    """
    Args:
        img: `(h, w)` (uint8)
    """
    image = Image.fromarray(img.astype("uint8"), mode="P")
    image.putpalette(sum(VOC_COLORMAP, []))
    return image

# image, gt = next(iter(train_dl))
# gt = gt[0, 0, ...]
# image = label_img_to_color(gt.numpy())
# image.show()