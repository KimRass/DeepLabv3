import torch

### Data
IMG_DIR = "/home/user/cv/voc2012/VOCdevkit/VOC2012/JPEGImages"
GT_DIR = "/home/user/cv/SegmentationClassAug"
# IMG_DIR = "/home/ubuntu/project/cv/voc2012/VOCdevkit/VOC2012/JPEGImages"
# GT_DIR = "/home/ubuntu/project/cv/SegmentationClassAug"
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
}
VOC_CLASSES = list(VOC_CLASS_COLOR.keys())[: -1]
N_CLASSES = len(VOC_CLASSES)
VOC_COLORS = list(VOC_CLASS_COLOR.values())
IMG_SIZE = 513
N_WORKERS = 4

### Optimizer
# "After training on the 'trainaug' set with 30K iterations and $initial learning rate = 0.007$,
# we then freeze batch normalization parameters, employ `output_stride = 8`, and train
# on the official PASCAL VOC 2012 trainval set for another 30K iterations
# and smaller $base learning rate = 0.001$."
INIT_LR = 0.007
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0004

## Training
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Using GPU.")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU.")
MULTI_GPU = True
# "Since large batch size is required to train batch normalization parameters, we employ `output_stride=16`
# and compute the batch normalization statistics with a batch size of 16.
# "The batch normalization parameters are trained with $decay = 0.9997$."
# BATCH_SIZE = 16 # In the paper
BATCH_SIZE = 14 # In my case (because of memory shortage)
N_WORKERS = 4
AUTOCAST = True
print(f"""AUTOCAST = {AUTOCAST}""")
N_STEPS = 300_000 # In the paper
N_PRINT_STEPS = 500
N_CKPT_STEPS = 6000
N_EVAL_STEPS = 3000

### Checkpoint
CKPT_PATH = None
STEP = None
TRANS_PHASE = None
RESOL_IDX = None
