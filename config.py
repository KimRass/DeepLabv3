### Data
IMG_DIR = "/home/user/cv/voc2012/VOCdevkit/VOC2012/JPEGImages"
GT_DIR = "/home/user/cv/SegmentationClassAug"
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
N_CLASSES = len(VOC_CLASSES)
VOC_COLORMAP = [
    (0, 0, 0),
    (128, 0, 0),
    (0, 128, 0),
    (128, 128, 0),
    (0, 0, 128),
    (128, 0, 128),
    (0, 128, 128),
    (128, 128, 128),
    (64, 0, 0),
    (192, 0, 0),
    (64, 128, 0),
    (192, 128, 0),
    (64, 0, 128),
    (192, 0, 128),
    (64, 128, 128),
    (192, 128, 128),
    (0, 64, 0),
    (128, 64, 0),
    (0, 192, 0),
    (128, 192, 0),
    (0, 64, 128),
    (255, 255, 255),
]
IMG_SIZE = 513
N_WORKERS = 4

### Optimizer
INIT_LR = 0.007
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0004

## Training
BATCH_SIZE = 16
N_WORKERS = 4
N_STEPS = 300_000 # In the paper
N_PRINT_STEPS = 500
N_CKPT_STEPS = 6000
N_EVAL_STEPS = 3000

### Checkpoint
CKPT_PATH = "/home/user/cv/deeplabv3_from_scratch/checkpoints/36000.pth"
STEP = None
TRANS_PHASE = None
RESOL_IDX = None
