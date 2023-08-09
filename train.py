# References:
    # https://github.com/PengtaoJiang/OAA-PyTorch/blob/master/deeplab-pytorch/libs/utils/lr_scheduler.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.cuda.amp import GradScaler
from pathlib import Path
from time import time
from contextlib import nullcontext
from tqdm.auto import tqdm

import config
from voc2012 import VOC2012Dataset
from model import DeepLabv3ResNet101
from loss import DeepLabLoss
from evaluate import PixelIoUByClass
from utils import get_lr, get_elapsed_time, save_checkpoint

print(f"""AUTOCAST = {config.AUTOCAST}""")
print(f"""N_WORKES = {config.N_WORKERS}""")
print(f"""BATCH_SIZE = {config.BATCH_SIZE}""")


def validate(val_dl, model, metric, device):
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
    print(f"""Average mIoU: {avg_miou:.4f}""")

    model.train()


model = DeepLabv3ResNet101(output_stride=16)
N_GPUS = torch.cuda.device_count()
if N_GPUS > 0:
    DEVICE = torch.device("cuda")
    model = model.to(DEVICE)
    if N_GPUS > 1 and config.MULTI_GPU:
        model = nn.DataParallel(model)

        print(f"""Using {N_GPUS} GPUs.""")
    else:
        print("Using single GPU.")
else:
    print("Using CPU.")

optim = SGD(
    params=model.parameters(),
    lr=config.INIT_LR,
    momentum=config.MOMENTUM,
    weight_decay=config.WEIGHT_DECAY,
)

scaler = GradScaler()

# Resume from checkpoint.
if config.CKPT_PATH is not None:
    ckpt = torch.load(config.CKPT_PATH, map_location=DEVICE)
    init_step = ckpt["step"]
    n_steps = ckpt["number_of_steps"]
    model.load_state_dict(ckpt["model"])
    optim.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])
else:
    init_step = 0
    n_steps = config.N_STEPS

train_ds = VOC2012Dataset(img_dir=config.IMG_DIR, gt_dir=config.GT_DIR, split="train")
train_dl = DataLoader(
    train_ds,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=config.N_WORKERS,
    pin_memory=True,
    drop_last=True,
)
train_di = iter(train_dl)

val_ds = VOC2012Dataset(img_dir=config.IMG_DIR, gt_dir=config.GT_DIR, split="val")
val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=config.N_WORKERS)

crit = DeepLabLoss()
metric = PixelIoUByClass()

### Train.
validate(val_dl=val_dl, model=model, metric=metric, device=DEVICE)

running_loss = 0
start_time = time()
for step in range(init_step + 1, n_steps + 1):
    try:
        image, gt = next(train_di)
    except StopIteration:
        train_di = iter(train_dl)
        image, gt = next(train_di)
    image = image.to(DEVICE)
    gt = gt.to(DEVICE)

    lr = get_lr(init_lr=config.INIT_LR, step=step, n_steps=n_steps)
    optim.param_groups[0]["lr"] = lr

    optim.zero_grad()

    with torch.autocast(
        device_type=DEVICE.type, dtype=torch.float16
    ) if config.AUTOCAST else nullcontext():
        pred = model(image)
        loss = crit(pred=pred, gt=gt)

    if config.AUTOCAST:
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
    else:
        loss.backward()
        optim.step()

    running_loss += loss.item()

    if step % config.N_PRINT_STEPS == 0:
        running_loss /= config.N_PRINT_STEPS
        print(f"""[ {step:,}/{n_steps:,} ][ {lr:4f} ][ {get_elapsed_time(start_time)} ]""", end="")
        print(f"""[ Loss: {running_loss:.4f} ]""")
        running_loss = 0

        start_time = time()

    if step % config.N_CKPT_STEPS == 0:
        save_checkpoint(
            step=step,
            n_steps=n_steps,
            model=model,
            optim=optim,
            scaler=scaler,
            save_path=Path(__file__).parent/f"""checkpoints/{step}.pth""",
        )
        print(f"""Saved checkpoint at step {step:,}/{n_steps:,}.""")

    ### Validate.
    if step % config.N_EVAL_STEPS == 0:
        validate(val_dl=val_dl, model=model, metric=metric)
