from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from pathlib import Path
import random

from utils import get_image_dataset_mean_and_std

IMG_SIZE = 513


class VOC2012Dataset(Dataset):
    def __init__(self, img_dir, gt_dir, split="train"):
        super().__init__()

        self.img_dir = Path(img_dir)
        self.split = split

        self.gts = list(Path(gt_dir).glob("*.png"))
        filenames = self.get_val_filenames()
        if split == "train":
            self.gts = [i for i in self.gts if i.stem not in filenames]
        elif split == "val":
            self.gts = [i for i in self.gts if i.stem in filenames]

    def get_val_filenames(self):
        val_txt_path = Path(self.img_dir).parent/"ImageSets/Segmentation/val.txt"
        with open(val_txt_path, mode="r") as f:
            filenames = [l.strip() for l in f.readlines()]
        return filenames

    def _transform(self, image, gt):
        if self.split == "train":
            # "Randomly left-right flipping"
            if random.random() > 0.5:
                image = TF.hflip(image)
                gt = TF.hflip(gt)

            # "We apply data augmentation by randomly scaling the input images (from 0.5 to 2.0)."
            w, h = gt.size
            scale = random.uniform(0.5, 2)
            size = (round(scale * h), round(scale * w))
            gt = TF.resize(gt, size=size, interpolation=Image.NEAREST)

            w, h = gt.size
            padding = (max(0, IMG_SIZE - w), max(0, IMG_SIZE - h))
            gt = TF.pad(gt, padding=padding, padding_mode="constant")
            # "We employ crop size to be $513$ during both training and test on PASCAL VOC 2012 dataset."
            t, l, h, w = T.RandomCrop.get_params(img=gt, output_size=(IMG_SIZE, IMG_SIZE))
            gt = TF.crop(gt, top=t, left=l, height=h, width=w)

            image = TF.resize(image, size=size)
            image = TF.pad(image, padding=padding, padding_mode="constant")
            image = TF.crop(image, top=t, left=l, height=h, width=w)

        elif self.split == "val":
            w, h = gt.size
            padding = (max(0, IMG_SIZE - w), max(0, IMG_SIZE - h))
            gt = TF.center_crop(gt, output_size=IMG_SIZE)

            image = TF.center_crop(image, output_size=IMG_SIZE)
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
        # `get_image_dataset_mean_and_std()`
        image = TF.normalize(image, mean=(0.452, 0.431, 0.399), std=(0.277, 0.273, 0.285))

        gt = TF.pil_to_tensor(gt).long()
        return image, gt


if __name__ == "__main__":
    img_dir = "/Users/jongbeomkim/Documents/datasets/voc2012/VOCdevkit/VOC2012/JPEGImages"
    gt_dir = "/Users/jongbeomkim/Documents/datasets/SegmentationClassAug"
    train_ds = VOC2012Dataset(img_dir=img_dir, gt_dir=gt_dir, split="train")
    image, gt = train_ds[random.choice(range(100))]
    image.shape, gt.shape

    gt = TF.pil_to_tensor(gt).long()
    gt[gt == 255] = 0
    gt = (gt.numpy() * 10 + 55).astype("uint8")[0]
    gt = Image.fromarray(gt)
    image.show(), gt.show()
