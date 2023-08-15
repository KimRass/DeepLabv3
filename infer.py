# References:
    # https://github.com/PengtaoJiang/OAA-PyTorch/blob/master/deeplab-pytorch/libs/utils/lr_scheduler.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from time import time
from tqdm.auto import tqdm

import config
from voc2012 import VOC2012Dataset
from model import DeepLabv3ResNet101
from evaluate import PixelIoUByClass
from utils import visualize_batched_image_and_gt

val_ds = VOC2012Dataset(img_dir=config.IMG_DIR, gt_dir=config.GT_DIR, split="val")
val_dl = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=config.N_WORKERS)

# metric = PixelIoUByClass()

model = DeepLabv3ResNet101(output_stride=16).to(config.DEVICE)
# if config.MULTI_GPU:
#     model = nn.DataParallel(model)
ckpt = torch.load(config.CKPT_PATH, map_location=config.DEVICE)
model.load_state_dict(ckpt["model"])

with torch.no_grad():
    for batch, (image, gt) in enumerate(tqdm(val_dl), start=1):
        image = image.to(config.DEVICE)
        gt = gt.to(config.DEVICE)

        pred = model(image)
        argmax = torch.argmax(pred, dim=1, keepdim=True)

        gt_vis = visualize_batched_image_and_gt(image=image, gt=gt, n_cols=4, alpha=0.7)
        gt_vis.save(f"""/Users/jongbeomkim/Downloads/deeplabv3_generated_images/{batch}_gt.jpg""")

        pred_vis = visualize_batched_image_and_gt(image=image, gt=argmax, n_cols=4, alpha=0.7)
        pred_vis.save(f"""/Users/jongbeomkim/Downloads/deeplabv3_generated_images/{batch}_pred.jpg""")
