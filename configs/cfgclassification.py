TASK = "classification"

GRID_SIZE = 5
SPLINE_ORDER = 3

BATCH_SIZE = 128
EPOCHS = 1
LR = 1e-3

IN_CHANNELS = 3
IMAGE_SIZE = 32
NUM_CLASSES = 100

MODEL_SIZE = "tiny[2+1]"

ARCHITECTURES = {
    "custom[6+1]": {
        "channels": [16, 16, 32, 32, 64, 64],
        "pools": [1, 3],
        "fc": [64, NUM_CLASSES],
        "ka_widths": [64, NUM_CLASSES],
    },
    "tiny[2+1]": {
        "channels": [16, 32],
        "pools": [0, 1],
        "fc": [64, NUM_CLASSES],
        "ka_widths": [64, NUM_CLASSES],
    },
    "small[3+2]": {
        "channels": [16, 32, 64],
        "pools": [0, 1],
        "fc": [64, 64, NUM_CLASSES],
        "ka_widths": [64, 64, NUM_CLASSES],
    },
}

MODEL_CFG = ARCHITECTURES[MODEL_SIZE]