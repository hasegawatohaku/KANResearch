import sys
import time

import torch
from torch.utils.data import DataLoader, random_split

from common.datasets import BUSISegmentationDataset

from configs.cfgany import DEVICE, TIME, SEED
from configs.cfgsegmentation import DATASET_ROOT, IMAGE_SIZE, BATCH_SIZE, IN_CHANNELS, LR, EPOCHS, GRID_SIZE, SPLINE_ORDER, ENCODER_CHANNELS, OUT_CHANNELS,  POOLS, KA_WIDTHS

from common.utils import Logger, count_parameters, set_seed, BCEDiceLoss

from archs.segmentation import build_models

sys.stdout = Logger(f"seg_{TIME}.log")
set_seed(SEED)

# ============================================================
# DATASET
# ============================================================

full_dataset = BUSISegmentationDataset(
    DATASET_ROOT,
    image_size=IMAGE_SIZE,
)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

split_generator = torch.Generator().manual_seed(SEED)

train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=split_generator,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=(DEVICE == "cuda"),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(DEVICE == "cuda"),
)

print("=" * 70)
print("BUSI SEGMENTATION DATASET")
print("=" * 70)
print("Total samples:", len(full_dataset))
print("Train samples:", len(train_dataset))
print("Val samples  :", len(val_dataset))
print(f"Image size   : {IN_CHANNELS} x {IMAGE_SIZE} x {IMAGE_SIZE}")
print("Output       : binary lesion mask")
print("Epochs       :", EPOCHS)

# ============================================================
# SEGMENTATION METRICS
# ============================================================
# Computes IoU and F1 for a batch of binary predictions.
# Predictions are thresholded at 0.5. Returns sum of IoU, sum of F1, and batch size so that the caller can accumulate averages.
@torch.no_grad()
def calculate_metrics(logits, target):
    """
    Metrics used in the ALL U-KAN paper:

        IoU = Intersection-over-Union
        F1  = F1 score

    Predictions are thresholded at 0.5.
    """

    prediction = (torch.sigmoid(logits) >= 0.5).float()

    intersection = (
        prediction * target
    ).sum(dim=(1, 2, 3))

    union = (
        prediction + target - prediction * target
    ).sum(dim=(1, 2, 3))

    prediction_sum = prediction.sum(dim=(1, 2, 3))
    target_sum = target.sum(dim=(1, 2, 3))

    iou = (intersection + 1e-6) / (union + 1e-6)

    f1 = (2.0 * intersection + 1e-6) / (prediction_sum + target_sum + 1e-6)

    return iou.sum().item(), f1.sum().item(), len(target)

# ============================================================
# TRAIN / VALIDATION
# ============================================================
# Full training loop for segmentation. After each logged epoch it calls
# evaluate_model and prints validation loss, IoU and F1.
def train_model(model):
    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR
    )

    criterion = BCEDiceLoss()

    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()

        total_loss = 0.0
        total_samples = 0

        for xb, yb in train_loader:
            xb = xb.to(
                DEVICE,
                non_blocking=True
            )
            yb = yb.to(
                DEVICE,
                non_blocking=True
            )

            optimizer.zero_grad()

            logits = model(xb)
            loss = criterion(logits, yb)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(xb)
            total_samples += len(xb)

        train_loss = total_loss / total_samples

        if epoch == 1 or epoch % 2 == 0:
            val_loss, val_iou, val_f1 = evaluate_model(
                model,
                criterion
            )

            print(
                f"epoch={epoch:3d} "
                f"loss={train_loss:.6f} "
                f"val_loss={val_loss:.6f} "
                f"IoU={100.0 * val_iou:.2f}% "
                f"F1={100.0 * val_f1:.2f}%"
            )

    elapsed = time.time() - start

    val_loss, val_iou, val_f1 = evaluate_model(
        model,
        criterion
    )

    return val_loss, val_iou, val_f1, elapsed

# Runs the model on the validation set without gradient tracking.
# Returns averaged validation loss, IoU and F1.
@torch.no_grad()
def evaluate_model(model, criterion):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    total_iou = 0.0
    total_f1 = 0.0
    metric_samples = 0

    for xb, yb in val_loader:
        xb = xb.to(
            DEVICE,
            non_blocking=True
        )
        yb = yb.to(
            DEVICE,
            non_blocking=True
        )

        logits = model(xb)
        loss = criterion(logits, yb)

        total_loss += loss.item() * len(xb)
        total_samples += len(xb)

        batch_iou, batch_f1, batch_size = calculate_metrics(
            logits,
            yb
        )

        total_iou += batch_iou
        total_f1 += batch_f1
        metric_samples += batch_size

    return (
        total_loss / total_samples,
        total_iou / metric_samples,
        total_f1 / metric_samples
    )

# ============================================================
# RUN
# ============================================================

print()
print("Device:", DEVICE)
print("Encoder channels:", ENCODER_CHANNELS)
print("Pools:", POOLS)
print("KA widths:", KA_WIDTHS)
print("Grid size:", GRID_SIZE)
print("Spline order:", SPLINE_ORDER)
print("N_spline:", GRID_SIZE + SPLINE_ORDER)
print("Loss: BCE + Dice")
print("Metrics: IoU + F1")

model_specs = build_models(
    KA_WIDTHS,
    IN_CHANNELS,
    OUT_CHANNELS,
    ENCODER_CHANNELS,
    POOLS,
    GRID_SIZE,
    SPLINE_ORDER
)

results = []

for name, builder in model_specs:
    # Construct only the current model to keep memory usage low.
    model = builder()

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    params = count_parameters(model)

    print("Total parameters:", params)

    val_loss, val_iou, val_f1, elapsed = train_model(model)

    print()
    print("Results:")
    print(f"Validation Loss = {val_loss:.6f}")
    print(f"IoU             = {100.0 * val_iou:.2f}%")
    print(f"F1              = {100.0 * val_f1:.2f}%")
    print(f"Time            = {elapsed:.2f} sec")

    results.append(
        (
            name,
            params,
            val_loss,
            val_iou,
            val_f1,
            elapsed
        )
    )


# ============================================================
# FINAL TABLE
# ============================================================

print()
print()
print("=" * 105)
print("FINAL RESULTS")
print("=" * 105)

print(
    f"{'Model':30s}"
    f"{'Params':>12s}"
    f"{'Val Loss':>14s}"
    f"{'IoU':>12s}"
    f"{'F1':>12s}"
    f"{'Time':>14s}"
)

print("-" * 105)

for name, params, val_loss, iou, f1, elapsed in results:
    print(
        f"{name:30s}"
        f"{params:12d}"
        f"{val_loss:14.6f}"
        f"{100.0 * iou:11.2f}%"
        f"{100.0 * f1:11.2f}%"
        f"{elapsed:13.2f}s"
    )