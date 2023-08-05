# References:
    # https://github.com/fregu856/deeplabv3/blob/master/utils/utils.py

from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
import torch
import torchvision.transforms as T
from time import time
from datetime import timedelta


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


# def label_img_to_color(img):
#     label_to_color = {
#         0: [128, 64,128],
#         1: [244, 35,232],
#         2: [ 70, 70, 70],
#         3: [102,102,156],
#         4: [190,153,153],
#         5: [153,153,153],
#         6: [250,170, 30],
#         7: [220,220,  0],
#         8: [107,142, 35],
#         9: [152,251,152],
#         10: [ 70,130,180],
#         11: [220, 20, 60],
#         12: [255,  0,  0],
#         13: [  0,  0,142],
#         14: [  0,  0, 70],
#         15: [  0, 60,100],
#         16: [  0, 80,100],
#         17: [  0,  0,230],
#         18: [119, 11, 32],
#         19: [81,  0, 81]
#     }
#     _, h, w = gt.shape
#     canvas = torch.zeros(size=(3, h, w), dtype=torch.long)
#     for label in gt.unique():
#         canvas[(gt == label)]
#         canvas[(gt == label)] = label_to_color[label]

#     img_height, img_width = img.shape

#     img_color = np.zeros((img_height, img_width, 3))
#     for row in range(img_height):
#         for col in range(img_width):
#             label = img[row, col]

#             img_color[row, col] = np.array(label_to_color[label])

#     return img_color