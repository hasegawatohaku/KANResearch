TASK = "regression"

GRID_SIZE = 5
SPLINE_ORDER = 3

BATCH_SIZE = 256
EPOCHS = 5
LR = 1e-3

MODEL_SIZE = "tiny"

ARCHITECTURES = {
    "tiny":   [8, 16, 16, 16, 16, 1],
    "small":  [8, 32, 32, 32, 32, 1],
    "medium": [8, 64, 64, 64, 64, 1],
}

WIDTHS = ARCHITECTURES[MODEL_SIZE]