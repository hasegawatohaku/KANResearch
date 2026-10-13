import torch

from  datetime import datetime as dt

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TIME = dt.now().strftime("%d.%m.%Y %H-%M-%S")