import os
import sys

import torch

import random
import numpy as np

# ============================================================
# SEED
# ============================================================
# Reproduces Python randomness.
# Forces cuDNN into deterministic mode when CUDA is available.
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ============================================================
# LOG STDOUT TO FILE
# ============================================================
# Redirects stdout to both the terminal and a log file.
class Logger(object):
    def __init__(self, filename):
        os.makedirs("./res", exist_ok=True)
        filepath = os.path.join("./res", filename)

        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def close(self):
        self.log.close()

# ============================================================
# PARAMETERS
# ============================================================
# Returns the total number of trainable parameters in a model.
def count_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
    )