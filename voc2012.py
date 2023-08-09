from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from pathlib import Path
import random

import config
from utils import (
    get_val_filenames,
    get_voc2012_trainaug_mean_and_std,
    visualize_batched_image_and_gt,
)


class VOC2012Dataset(Dataset):
    def __init__(self, img_dir, gt_dir, split="train"):
        super().__init__()

        self.img_dir = Path(img_dir)
        self.split = split

        self.gts = list(Path(gt_dir).glob("*.png"))
        filenames = get_val_filenames(self.img_dir)
        if split == "train":
            self.gts = [i for i in self.gts if i.stem not in filenames]
        elif split == "val":
            self.gts = [i for i in self.gts if i.stem in filenames]

    def _randomly_adjust_b_and_s(self, image):
        image = TF.adjust_brightness(image, random.uniform(0.5, 1.5))
        image = TF.adjust_saturation(image, random.uniform(0.5, 1.5))
        return image

    # "Randomly left-right flipping"
    def _randomly_flip_horizontally(self, image, gt, p=0.5):
        if random.random() > 1 - p:
            image = TF.hflip(image)
            gt = TF.hflip(gt)
        return image, gt

    def _randomly_scale(self, image, gt): # Not in the paper
        # "We apply data augmentation by randomly scaling the input images (from 0.5 to 2.0)."
        w, h = gt.size
        scale = random.uniform(0.5, 2)
        size = (round(scale * h), round(scale * w))
        gt = TF.resize(gt, size=size, interpolation=Image.NEAREST)
        image = TF.resize(image, size=size)
        return image, gt

    def _randomly_crop(self, image, gt):
        w, h = gt.size
        padding = (max(0, config.IMG_SIZE - w), max(0, config.IMG_SIZE - h))
        gt = TF.pad(gt, padding=padding, padding_mode="constant")
        # "We employ crop size to be $513$ during both training and test on PASCAL VOC 2012 dataset."
        t, l, h, w = T.RandomCrop.get_params(img=gt, output_size=(config.IMG_SIZE, config.IMG_SIZE))
        gt = TF.crop(gt, top=t, left=l, height=h, width=w)

        image = TF.pad(image, padding=padding, padding_mode="constant")
        image = TF.crop(image, top=t, left=l, height=h, width=w)
        return image, gt

    def _transform(self, image, gt):
        if self.split == "train": # 10,582 images and labels
            image = self._randomly_adjust_b_and_s(image)
            image, gt = self._randomly_flip_horizontally(image=image, gt=gt)
            image, gt = self._randomly_scale(image=image, gt=gt)
            image, gt = self._randomly_crop(image=image, gt=gt)

        elif self.split == "val": # 1,449 images and labels
            gt = TF.center_crop(gt, output_size=config.IMG_SIZE)
            image = TF.center_crop(image, output_size=config.IMG_SIZE)
        return image, gt

    def __len__(self):
        return len(self.gts)

    def __getitem__(self, idx):
        gt_path = self.gts[idx]
        gt = Image.open(gt_path)

        img_path = f"""{self.img_dir/gt_path.stem}.jpg"""
        image = Image.open(img_path).convert("RGB")

        image, gt = self._transform(image=image, gt=gt)

        image = TF.to_tensor(image)
        # `get_voc2012_trainaug_mean_and_std`
        image = TF.normalize(image, mean=(0.457, 0.437, 0.404), std=(0.275, 0.271, 0.284))

        gt = TF.pil_to_tensor(gt).long()
        return image, gt


if __name__ == "__main__":
    img_dir = "/Users/jongbeomkim/Documents/datasets/voc2012/VOCdevkit/VOC2012/JPEGImages"
    gt_dir = "/Users/jongbeomkim/Documents/datasets/SegmentationClassAug"
    train_ds = VOC2012Dataset(img_dir=img_dir, gt_dir=gt_dir, split="train")
    train_dl = DataLoader(
        train_ds, batch_size=16, shuffle=True, num_workers=0, pin_memory=True, drop_last=True
    )
    cnt = 0
    for _ in range(10):
        image, gt = next(iter(train_dl))
        vis = visualize_batched_image_and_gt(image, gt, n_cols=4, alpha=0.7)
        vis.show()
        vis.save(f"""/Users/jongbeomkim/Desktop/workspace/deeplabv3_from_scratch/input_images/{cnt}.jpg""")
        cnt += 1
