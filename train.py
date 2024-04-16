# References:
    # https://github.com/PengtaoJiang/OAA-PyTorch/blob/master/deeplab-pytorch/libs/utils/lr_scheduler.py

import torch
from torch.utils.data import DataLoader
from torch.optim import SGD
from pathlib import Path
from time import time
import contextlib
from tqdm import tqdm
import argparse

from voc2012 import VOC2012Dataset
from model import ResNet101DeepLabv3
from utils import get_elapsed_time, get_device, set_seed, get_grad_scaler


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, required=False, default=123)
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--gt_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--n_cpus", type=int, required=True)
    parser.add_argument("--n_steps", type=int, required=False, default=30_000) # In the paper
    parser.add_argument("--resume_from", type=str, required=False)
    ### Optimizer
    parser.add_argument("--init_lr", type=float, required=False, default=0.007)
    parser.add_argument("--momentum", type=float, required=False, default=0.9)
    parser.add_argument("--weight_decay", type=float, required=False, default=0.0004)
    ### Logging
    parser.add_argument("--log_every", type=int, required=False, default=500)
    parser.add_argument("--save_every", type=int, required=False, default=6000)
    parser.add_argument("--val_every", type=int, required=False, default=3000)

    args = parser.parse_args()

    args_dict = vars(args)
    new_args_dict = dict()
    for k, v in args_dict.items():
        new_args_dict[k.upper()] = v
    args = argparse.Namespace(**new_args_dict)
    return args


class Trainer(object):
    """
    "After training on the 'trainaug' set with 30K iterations and $initial learning rate = 0.007$,
        we then freeze batch normalization parameters, employ `output_stride = 8`, and train
        on the official PASCAL VOC 2012 trainval set for another 30K iterations
        and smaller $base learning rate = 0.001$."
    "Since large batch size is required to train batch normalization parameters, we employ `output_stride=16`
        and compute the batch normalization statistics with a batch size of 16."
    "The batch normalization parameters are trained with $decay = 0.9997$."
    """
    def __init__(
        self,
        train_dl,
        val_dl,
        save_dir,
        init_lr,
        n_steps,
        device,
    ):
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.save_dir = Path(save_dir)
        self.init_lr = init_lr
        self.n_steps = n_steps
        self.device = device

    def get_lr(self, step, power=0.9):
        """
        "We employ a 'poly' learning rate policy where the initial learning rate is multiplied
            by $(1 - \frac{iter}{max{\_}iter})^{power}$ with $power = 0.9$."
        """
        lr = self.init_lr * (1 - (step / self.n_steps)) ** power
        return lr

    @staticmethod
    def update_lr(lr, optim):
        optim.param_groups[0]["lr"] = lr

    def train_for_one_step(self, image, gt, step, model, optim, scaler):
        image = image.to(self.device)
        gt = gt.to(self.device)

        lr = self.get_lr(step=step)
        self.update_lr(lr=lr, optim=optim)

        with torch.autocast(
            device_type=self.device.type, dtype=torch.float16,
        ) if self.device.type == "cuda" else contextlib.nullcontext():
            loss = model.get_loss(image=image, gt=gt)
        optim.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
            optim.step()
        # loss = torch.randn(size=(1,), device=self.device)
        return loss

    def save_checkpoint(self, step, model, optim, scaler, max_avg_miou, save_path):
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "step": step,
            "number_of_steps": self.n_steps,
            "model": model.state_dict(),
            "optimizer": optim.state_dict(),
            "maximum_average_mean_iou": max_avg_miou,
        }
        if scaler is not None:
            ckpt["scaler"] = scaler.state_dict()
        torch.save(ckpt, str(save_path))

    def save_model_params(self, model, save_path):
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), str(save_path))

    @torch.inference_mode()
    def validate(self, model):
        model.eval()
        cum_miou = 0
        pbar = tqdm(self.val_dl)
        for image, gt in pbar:
            pbar.set_description("Validating...")

            image = image.to(self.device)
            gt = gt.to(self.device)

            pred = model(image)
            ious = model.get_pixel_iou_by_cls(pred=pred, gt=gt)
            miou = sum(ious.values()) / len(ious)

            cum_miou += miou
        avg_miou = cum_miou / len(self.val_dl)
        # import random
        # avg_miou = random.random()
        model.train()
        return avg_miou

    def train(
        self,
        init_step,
        max_avg_miou,
        model,
        optim,
        scaler,
        log_every,
        save_every,
        val_every,
    ):
        train_di = iter(self.train_dl)

        cum_loss = 0
        start_time = time()
        pbar = tqdm(range(init_step + 1, self.n_steps + 1), leave=False)
        for step in pbar:
            pbar.set_description("Training...")

            try:
                image, gt = next(train_di)
            except StopIteration:
                train_di = iter(self.train_dl)
                image, gt = next(train_di)
            loss = self.train_for_one_step(
                step=step,
                model=model,
                optim=optim,
                scaler=scaler,
                image=image,
                gt=gt,
            )
            cum_loss += loss.item()

            if step % log_every == 0:
                cum_loss /= log_every
                log = f"[ {get_elapsed_time(start_time)} ]"
                log += f"[ {step:,}/{self.n_steps:,} ]\n"
                log += f"[ Loss: {cum_loss:.4f} ]"
                print(log)
                cum_loss = 0
                start_time = time()

            if step % save_every == 0:
                self.save_checkpoint(
                    step=step,
                    model=model,
                    optim=optim,
                    scaler=scaler,
                    max_avg_miou=max_avg_miou,
                    save_path=self.save_dir/f"step={step}.pth",
                )
                log = f"[ {step:,}/{self.n_steps:,} ][ Checkpoint saved. ]"
                print(log)

            if step % val_every == 0:
                avg_miou = self.validate(model=model)
                if avg_miou > max_avg_miou:
                    self.save_model_params(
                        model=model,
                        save_path=self.save_dir/f"step={step}-avg_miou={avg_miou:.4f}.pth",
                    )
                    max_avg_miou = avg_miou
                log = f"[ {step:,}/{self.n_steps:,} ]"
                log += f"[ Average mIoU: {avg_miou:.4f} | {max_avg_miou:.4f} ]"
                print(log)


def main():
    DEVICE = get_device()
    print(f"[ DEVICE: {DEVICE} ]")
    args = get_args()
    set_seed(args.SEED)

    model = ResNet101DeepLabv3(output_stride=16).to(DEVICE)
    optim = SGD(
        params=model.parameters(),
        lr=args.INIT_LR,
        momentum=args.MOMENTUM,
        weight_decay=args.WEIGHT_DECAY,
    )
    scaler = get_grad_scaler(device=DEVICE)

    if args.RESUME_FROM is not None:
        ckpt = torch.load(args.RESUME_FROM, map_location=DEVICE)
        init_step = ckpt["step"]
        n_steps = ckpt["number_of_steps"]
        max_avg_miou = ckpt["maximum_average_mean_iou"]
        model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optimizer"])
        if scaler is not None:
            scaler.load_state_dict(ckpt["scaler"])
        print(f"Resume training from {init_step:,}/{n_steps:,} steps")
    else:
        init_step = 0
        n_steps = args.N_STEPS
        max_avg_miou = 0

    train_ds = VOC2012Dataset(img_dir=args.IMG_DIR, gt_dir=args.GT_DIR, split="train")
    train_dl = DataLoader(
        train_ds,
        batch_size=args.BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        num_workers=args.N_CPUS,
    )
    val_ds = VOC2012Dataset(img_dir=args.IMG_DIR, gt_dir=args.GT_DIR, split="val")
    val_dl = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
        num_workers=args.N_CPUS,
    )

    trainer = Trainer(
        train_dl=train_dl,
        val_dl=val_dl,
        save_dir=args.SAVE_DIR,
        init_lr=args.INIT_LR,
        n_steps=n_steps,
        device=DEVICE,
    )
    trainer.train(
        init_step=init_step,
        max_avg_miou=max_avg_miou,
        model=model,
        optim=optim,
        scaler=scaler,
        log_every=args.LOG_EVERY,
        save_every=args.SAVE_EVERY,
        val_every=args.VAL_EVERY,
    )

if __name__ == "__main__":
    main()
