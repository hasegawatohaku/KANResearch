import sys
import time

from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from common.datasets import BUSISegmentationDataset

from configs.cfgany import DEVICE, TIME, SEED
from configs.cfgsegmentation import (DATASET_ROOT, 
                                     IMAGE_SIZE, 
                                     BATCH_SIZE, 
                                     IN_CHANNELS, 
                                     LR, EPOCHS, 
                                     GRID_SIZE, 
                                     SPLINE_ORDER, 
                                     ENCODER_CHANNELS, 
                                     OUT_CHANNELS,  
                                     POOLS, 
                                     KA_WIDTHS)
from configs.cfglut import (
                     KANELE_LUT_NAME,
                     KANELE_BITS, 
                     KANELE_EXPORT_LUT, 
                     KANELE_INPUT_RANGE, 
                     KANELE_OUTPUT_RANGE, 
                     KANELE_QMAX, 
                     KANELE_QMIN, 
                     KANELE_TABLE_FRAC_BITS,
                     KANELE_TARGET_SPARSITY,
                     KANELE_PRUNING_START_EPOCH,
                     KANELE_CALIBRATION_IMAGES,
                     KANELE_CHUNK_ELEMENTS)

from train.testlut.lut_inf_model import build_kanele_lut_model, kanele_lut_widths, export_kanele_lut

from common.loss import BCEDiceLoss
from common.utils import Logger, count_parameters, set_seed
from common.pruning import count_kanele_connections, make_kanele_epoch_end

from archs.segmentation import build_models

sys.stdout = Logger(f"seg_{TIME}.log")

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
def train_model(model, epoch_end):
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

        if epoch_end is not None:
            epoch_end(epoch)

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

@torch.no_grad()
def compare_qat_lut(model, lut_model):
    """
    Agreement of the QAT model and its LUT model on the validation set.

    Both models sum the same integer table entries and apply the same
    integer rounding and saturation, so their outputs must be identical.
    A non-zero difference indicates an error.
    """

    model.eval()
    lut_model.eval()

    max_difference = 0.0
    different_logits = 0
    different_pixels = 0
    pixels = 0

    for xb, _ in val_loader:
        xb = xb.to(
            DEVICE,
            non_blocking=True
        )

        qat_logits = model(xb)
        lut_logits = lut_model(xb)

        difference = (qat_logits - lut_logits).abs()

        max_difference = max(max_difference, difference.max().item())
        different_logits += int((difference > 0).sum().item())

        # Same decision rule as calculate_metrics().
        qat_mask = torch.sigmoid(qat_logits) >= 0.5
        lut_mask = torch.sigmoid(lut_logits) >= 0.5

        different_pixels += int((qat_mask != lut_mask).sum().item())
        pixels += qat_logits.numel()

    return (
        max_difference,
        different_logits / pixels,
        different_pixels / pixels
    )


def run_kanele_lut_inference(model, params, chunk_elements):
    """Build the LUT model, evaluate it and compare it with the QAT model."""

    criterion = BCEDiceLoss()

    start = time.time()
    lut_model = build_kanele_lut_model(model, chunk_elements)
    build_time = time.time() - start

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    start = time.time()

    lut_loss, lut_iou, lut_f1 = evaluate_model(
        lut_model,
        criterion
    )

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    inference_time = time.time() - start

    max_difference, logit_mismatch, pixel_mismatch = compare_qat_lut(
        model,
        lut_model
    )

    active, total = count_kanele_connections(model)
    entry_bits, accumulator_bits = kanele_lut_widths(lut_model)

    print()
    print("Pruning:")
    print(f"Active connections = {active}/{total} "
          f"({100.0 * (total - active) / total:.2f}% pruned)")

    print()
    print(f"LUT build time  = {build_time:.2f} sec")
    print(f"Inference time  = {inference_time:.2f} sec "
          f"(validation set, including data loading)")
    print(f"LUT size        = {active} active L-LUTs x "
          f"{1 << KANELE_BITS} entries")
    print(f"Widths          = table entries up to {entry_bits} bits, "
          f"accumulators up to {accumulator_bits} bits "
          f"(fractional bits F = {KANELE_TABLE_FRAC_BITS})")
    print(f"Max |QAT - LUT| logit difference  = {max_difference:.8g}")
    print(f"Logits different from QAT model   = {100.0 * logit_mismatch:.4f}%")
    print(f"Mask pixels different from QAT    = {100.0 * pixel_mismatch:.4f}%")

    if KANELE_EXPORT_LUT:
        path = Path(f"kanele_segmentation_lut {TIME}.json")
        export_kanele_lut(lut_model, path, ENCODER_CHANNELS, POOLS, KA_WIDTHS)
        print("Exported LUT:", path.resolve())

    return (
        KANELE_LUT_NAME,
        params,
        lut_loss,
        lut_iou,
        lut_f1,
        inference_time
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

print(
    f"KANELE:\n\t- bits={KANELE_BITS},\n"
    f"\t- activation range=[{KANELE_QMIN}, {KANELE_QMAX}]\n"
    f"\t- input range=[{KANELE_INPUT_RANGE[0]}, {KANELE_INPUT_RANGE[1]}]\n"
    f"\t- output range=[{KANELE_OUTPUT_RANGE[0]}, {KANELE_OUTPUT_RANGE[1]}]\n"
    f"\t- table fractional bits={KANELE_TABLE_FRAC_BITS}\n"
    f"\t- target sparsity={100.0 * KANELE_TARGET_SPARSITY:.1f}%\n"
    f"\t- pruning from epoch {KANELE_PRUNING_START_EPOCH}"
)

model_specs = build_models(
    KA_WIDTHS,
    IN_CHANNELS,
    OUT_CHANNELS,
    ENCODER_CHANNELS,
    POOLS,
    GRID_SIZE,
    SPLINE_ORDER,
    KANELE_BITS,
    KANELE_INPUT_RANGE,
    (KANELE_QMIN, KANELE_QMAX),
    KANELE_OUTPUT_RANGE,
    KANELE_TABLE_FRAC_BITS
)

results = []

for name, builder, is_kanele in model_specs:
    set_seed(SEED)
    
    # Construct only the current model to keep memory usage low.
    model = builder()

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    params = count_parameters(model)

    print("Total parameters:", params)

    # Only ALL U-KAN + KANELE uses the epoch-end hook (pruning).
    epoch_end = make_kanele_epoch_end(model, 
                                    KANELE_CALIBRATION_IMAGES, 
                                    train_dataset, 
                                    KANELE_TARGET_SPARSITY, 
                                    KANELE_PRUNING_START_EPOCH, 
                                    EPOCHS,
                                    BATCH_SIZE) if is_kanele else None


    val_loss, val_iou, val_f1, elapsed = train_model(model, epoch_end)

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

    # LUT inference of the trained ALL U-KAN + KANELE model.
    if is_kanele:
        results.append(
            run_kanele_lut_inference(model, params, KANELE_CHUNK_ELEMENTS)
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