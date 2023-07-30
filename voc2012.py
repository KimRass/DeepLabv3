from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random

from utils import get_image_dataset_mean_and_std


VOC_CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
]

IMG_SIZE = 513


class VOC2012Dataset(Dataset):
    def __init__(self, root_dir):
        super().__init__()

        self.labels = list((Path(root_dir)/"SegmentationClass").glob("*.png"))

    def _transform(self, image, label):
        # "Randomly left-right flipping"
        if random.random() > 0.5:
            image = TF.hflip(image)
            label = TF.hflip(label)

        # "We apply data augmentation by randomly scaling the input images (from 0.5 to 2.0)."
        w, h = label.size
        scale = random.uniform(0.5, 2)
        size = (round(scale * h), round(scale * w))
        label = TF.resize(label, size=size)
        w, h = label.size
        padding = (max(0, IMG_SIZE - w), max(0, IMG_SIZE - h))
        label = TF.pad(label, padding=padding, padding_mode="edge")
        # "We employ crop size to be $513$ during both training and test on PASCAL VOC 2012 dataset."
        t, l, h, w = T.RandomCrop.get_params(img=label, output_size=(IMG_SIZE, IMG_SIZE))
        label = TF.crop(label, top=t, left=l, height=h, width=w)
        
        image = TF.resize(image, size=size)
        image = TF.pad(image, padding=padding, padding_mode="edge")
        image = TF.crop(image, top=t, left=l, height=h, width=w)
        return image, label

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label_path = self.labels[idx]
        label = Image.open(label_path)

        img_path = f"""{label_path.parent.parent/"JPEGImages"/label_path.stem}.jpg"""
        image = Image.open(img_path).convert("RGB")

        image, label = self._transform(image=image, label=label)
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=(0.452, 0.431, 0.399), std=(0.277, 0.273, 0.285))
        label = TF.pil_to_tensor(label)
        return image, label


if __name__ == "__main__":
    root_dir = "/Users/jongbeomkim/Documents/datasets/voc2012/VOCdevkit/VOC2012"
    idx = 3
    ds = VOC2012Dataset(root_dir=root_dir)
    image, label = ds[2]
    label
