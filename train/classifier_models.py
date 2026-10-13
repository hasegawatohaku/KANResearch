import sys
import time

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from configs.cfgany import DEVICE, TIME
from configs.cfgclassification import EPOCHS, LR, MODEL_SIZE, BATCH_SIZE, IN_CHANNELS, IMAGE_SIZE, NUM_CLASSES, MODEL_CFG, GRID_SIZE, SPLINE_ORDER

from common.utils import Logger, count_parameters

from archs.classification import build_models

sys.stdout = Logger(f"cls_{TIME}.log")

# ============================================================
# DATASET
# ============================================================
# Reference values for normalization
# https://www.ricardodecal.com/guides/use-these-normalization-values-for-torchvision-datasets/
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
    (0.5071, 0.4867, 0.4408),
    (0.2675, 0.2565, 0.2761))])

train_dataset = datasets.CIFAR100(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.CIFAR100(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=(DEVICE == "cuda")
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(DEVICE == "cuda")
)

print("=" * 60)
print(train_dataset.__class__.__name__)
print("=" * 60)

print("Train samples:", len(train_dataset))
print("Test samples :", len(test_dataset))
print(f"Image size   : {IN_CHANNELS} x {IMAGE_SIZE} x {IMAGE_SIZE}")
print("Classes      :", NUM_CLASSES)
print("Epochs       :", EPOCHS)

# ============================================================
# TRAIN / TEST
# ============================================================
# Trains a classification model for `EPOCHS` epochs with Adam + CrossEntropy.
# Prints loss/accuracy every other epoch and returns test loss, test accuracy, and elapsed time.
def train_model(model):
    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR
    )
    criterion = nn.CrossEntropyLoss()

    start = time.time()

    model.train()

    for epoch in range(1, EPOCHS + 1):
        #model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for xb, yb in train_loader:

            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            logits = model(xb)

            loss = criterion(logits, yb)

            loss.backward()

            optimizer.step()

            total_loss += loss.item() * len(xb)
            total_correct += (logits.argmax(dim=1) == yb).sum().item()
            total_samples += len(xb)

        train_loss = total_loss / total_samples
        train_accuracy = 100.0 * total_correct / total_samples

        if epoch == 1 or epoch % 2 == 0:
            print(
                f"epoch={epoch:3d} "
                f"loss={train_loss:.6f} "
                f"accuracy={train_accuracy:.2f}%"
            )

    elapsed = time.time() - start

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(
                DEVICE,
                non_blocking=True,
            )
            yb = yb.to(
                DEVICE,
                non_blocking=True,
            )

            logits = model(xb)

            loss = criterion(
                logits,
                yb,
            )

            total_loss += loss.item() * len(xb)
            total_correct += (logits.argmax(dim=1) == yb).sum().item()
            total_samples += len(xb)

    test_loss = total_loss / total_samples
    test_accuracy = 100.0 * total_correct / total_samples

    return test_loss, test_accuracy, elapsed

# ============================================================
# RUN
# ============================================================

print()
print("Device:", DEVICE)
print("Model size:", MODEL_SIZE)
print("Channels:", MODEL_CFG["channels"])
print("KA widths:", MODEL_CFG["ka_widths"])
print("FC width:", MODEL_CFG["fc"])
print("Number of classes:", NUM_CLASSES)
print("Grid size:", GRID_SIZE)
print("Spline order:", SPLINE_ORDER)
print("N_spline:", GRID_SIZE + SPLINE_ORDER)

models = build_models(
    IN_CHANNELS,
    MODEL_CFG,
    GRID_SIZE,
    SPLINE_ORDER,
    IMAGE_SIZE
)

results = []

for name, model in models:
    print()
    print("=" * 60)
    print(name)
    print("=" * 60)

    params = count_parameters(model)

    print("Total parameters:", params)

    test_loss, test_accuracy, elapsed = train_model(model)

    print()
    print("Results:")
    print(f"Test Loss     = {test_loss:.6f}")
    print(f"Test Accuracy = {test_accuracy:.2f}%")
    print(f"Time          = {elapsed:.2f} sec")

    results.append(
        (
            name,
            params,
            test_loss,
            test_accuracy,
            elapsed,
        )
    )


# ============================================================
# FINAL TABLE
# ============================================================

print()
print()
print("=" * 85)
print("FINAL RESULTS")
print("=" * 85)

print(
    f"{'Model':30s}"
    f"{'Params':>12s}"
    f"{'Test Loss':>15s}"
    f"{'Accuracy':>13s}"
    f"{'Time':>14s}"
)

print("-" * 85)

for name, params, test_loss, accuracy, elapsed in results:
    print(
        f"{name:30s}"
        f"{params:12d}"
        f"{test_loss:15.6f}"
        f"{accuracy:12.2f}%"
        f"{elapsed:13.2f}s"
    )