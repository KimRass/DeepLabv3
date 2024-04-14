import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from PIL import Image
from pathlib import Path
import random
import numpy as np
from tqdm import tqdm

from utils import visualize_batched_image_and_gt


class VOC2012Dataset(Dataset):
    # `get_mean_and_std`
    def __init__(
        self,
        img_dir,
        gt_dir,
        img_size=513,
        mean=(0.457, 0.437, 0.404),
        std=(0.275, 0.271, 0.284),
        split="train",
    ):
        super().__init__()

        self.img_dir = Path(img_dir)
        self.gt_dir = Path(gt_dir)
        self.img_size = img_size
        val_filenames = self.get_val_filenames()
        if mean is None and std is None:
            self.mean, self.std = self.get_mean_and_std(val_filenames)
        else:
            self.mean = mean
            self.std = std
        self.split = split

        self.gts = list(self.gt_dir.glob("*.png"))
        if split == "train":
            self.gts = [i for i in self.gts if i.stem not in val_filenames]
        elif split == "val":
            self.gts = [i for i in self.gts if i.stem in val_filenames]

        self.val_transform = self.get_val_transform(
            img_size=img_size, mean=self.mean, std=self.std,
        )

    def get_val_filenames(self):
        val_txt_path = self.img_dir.parent/"ImageSets/Segmentation/val.txt"
        with open(val_txt_path, mode="r") as f:
            filenames = [l.strip() for l in f.readlines()]
        return filenames

    def get_mean_and_std(self, val_filenames):
        cnt = 0
        sum_rgb = 0
        sum_rgb_square = 0
        sum_resol = 0
        for gt_path in tqdm(list(self.gt_dir.glob("*.png"))):
            if gt_path.stem in val_filenames:
                continue
            cnt += 1

            img_path = (self.img_dir/gt_path.stem).with_suffix(".jpg")
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

    @classmethod
    def get_val_transform(cls, img_size, mean, std):
        return A.Compose(
            [
                A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_AREA),
                A.PadIfNeeded(
                    min_height=img_size,
                    min_width=img_size,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=tuple([int(i * 255) for i in mean]),
                ),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ],
        )

    def _randomly_adjust_b_and_s(self, image):
        image = TF.adjust_brightness(image, random.uniform(0.5, 1.5))
        image = TF.adjust_saturation(image, random.uniform(0.5, 1.5))
        return image

    def _randomly_flip_horizontally(self, image, gt, p=0.5):
        """
        "Randomly left-right flipping"
        """
        if random.random() > 1 - p:
            image = TF.hflip(image)
            gt = TF.hflip(gt)
        return image, gt

    def _randomly_scale(self, image, gt):
        """
        "We apply data augmentation by randomly scaling the input images (from 0.5 to 2.0)."
        """
        w, h = gt.size
        scale = random.uniform(0.5, 2)
        size = (round(scale * h), round(scale * w))
        gt = TF.resize(gt, size=size, interpolation=Image.NEAREST)
        image = TF.resize(image, size=size)
        return image, gt

    def _randomly_crop(self, image, gt):
        """
        "We employ crop size to be $513$ during both training and test on PASCAL VOC 2012
            dataset."
        """
        w, h = gt.size
        padding = (max(0, self.img_size - w), max(0, self.img_size - h))
        gt = TF.pad(gt, padding=padding, padding_mode="constant")
        t, l, h, w = T.RandomCrop.get_params(img=gt, output_size=(self.img_size, self.img_size))
        gt = TF.crop(gt, top=t, left=l, height=h, width=w)

        image = TF.pad(image, padding=padding, padding_mode="constant")
        image = TF.crop(image, top=t, left=l, height=h, width=w)
        return image, gt

    def _transform(self, image, gt):
        if self.split == "train":
            image = self._randomly_adjust_b_and_s(image)
            image, gt = self._randomly_flip_horizontally(image=image, gt=gt)
            image, gt = self._randomly_scale(image=image, gt=gt)
            image, gt = self._randomly_crop(image=image, gt=gt)
            image = TF.to_tensor(image)
            image = TF.normalize(image, mean=self.mean, std=self.std)
            gt = TF.pil_to_tensor(gt)

        elif self.split == "val":
            transformed = self.val_transform(image=np.array(image), mask=np.array(gt))
            image = transformed["image"]
            gt = transformed["mask"][None, ...]
        return image, gt.long()

    def __len__(self):
        return len(self.gts)

    def __getitem__(self, idx):
        gt_path = self.gts[idx]
        gt = Image.open(gt_path)
        image = Image.open(f"{self.img_dir/gt_path.stem}.jpg").convert("RGB")
        image, gt = self._transform(image=image, gt=gt)
        return image, gt


if __name__ == "__main__":
    img_dir = "/Users/jongbeomkim/Documents/datasets/voc2012/VOCdevkit/VOC2012/JPEGImages"
    gt_dir = "/Users/jongbeomkim/Documents/datasets/SegmentationClassAug"
    train_ds = VOC2012Dataset(img_dir=img_dir, gt_dir=gt_dir, split="train")
    train_dl = DataLoader(
        train_ds, batch_size=4, shuffle=True, num_workers=0, pin_memory=True, drop_last=True
    )
    image, gt = next(iter(train_dl))
    visualize_batched_image_and_gt(image, gt, n_cols=4, alpha=0.7)
