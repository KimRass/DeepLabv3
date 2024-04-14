### Data
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
} # DO NOT MODIFY
VOC_CLASSES = list(VOC_CLASS_COLOR.keys())[: -1] # DO NOT MODIFY
N_CLASSES = len(VOC_CLASSES) # DO NOT MODIFY
VOC_COLORS = list(VOC_CLASS_COLOR.values()) # DO NOT MODIFY
IMG_SIZE = 513 # DO NOT MODIFY
