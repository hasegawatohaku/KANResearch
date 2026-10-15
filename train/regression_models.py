import sys
import time
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from configs.cfgany import DEVICE, TIME, SEED
from configs.cfgregression import MODEL_SIZE, BATCH_SIZE, LR, EPOCHS, WIDTHS, GRID_SIZE, SPLINE_ORDER

from common.utils import Logger, count_parameters, set_seed

from archs.regression import build_models

sys.stdout = Logger(f"reg_{TIME}.log")

# ============================================================
# DATASET
# ============================================================

print("=" * 60)
print("CALIFORNIA HOUSING")
print("=" * 60)

data = fetch_california_housing()

x = data.data.astype(np.float32)
y = data.target.astype(np.float32).reshape(-1, 1)

print("Samples:", len(x))
print("Features:", x.shape[1])

# train / test split
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=SEED,
)

# normalize features
x_scaler = StandardScaler()
x_train = x_scaler.fit_transform(x_train).astype(np.float32)
x_test = x_scaler.transform(x_test).astype(np.float32)

# normalize target
y_scaler = StandardScaler()
y_train = y_scaler.fit_transform(y_train).astype(np.float32)
y_test = y_scaler.transform(y_test).astype(np.float32)

# tensors
x_train = torch.tensor(x_train)
x_test = torch.tensor(x_test)

y_train = torch.tensor(y_train)
y_test = torch.tensor(y_test)

# DataLoader
train_loader = DataLoader(
    TensorDataset(
        x_train,
        y_train,
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=(DEVICE == "cuda"),
)

# ============================================================
# TRAIN / TEST
# ============================================================
# Trains and evaluates a regression model on the California Housing dataset.
# Logs MSE every `log_every` epochs. Returns metrics on the test set:
# MSE and RMSE on the original target scale, Adjusted R², and a threshold accuracy (|pred - true| < epsilon).
def train_model(model):

    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
    )

    criterion = nn.MSELoss()

    start = time.time()

    model.train()

    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0

        for xb, yb in train_loader:

            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            pred = model(xb)

            loss = criterion(pred, yb)

            loss.backward()

            optimizer.step()

            total_loss += loss.item() * len(xb)

        if (epoch == 1 or epoch % 10 == 0):

            avg_loss = total_loss / len(x_train)

            print(
                f"epoch={epoch:3d} "
                f"loss={avg_loss:.6f}"
            )

    elapsed = time.time() - start

    # Test
    model.eval()


    with torch.no_grad():

        x_test_device = x_test.to(
            DEVICE,
            non_blocking=True,
        )
        y_test_device = y_test.to(
            DEVICE,
            non_blocking=True,
        )

        pred = model(x_test_device)

        # MSE
        mse_normalized = torch.mean((pred - y_test_device) ** 2).item()

        # RMSE
        rmse_normalized = (mse_normalized ** 0.5)

        # Adjusted R²
        ss_res = torch.sum((y_test_device - pred) ** 2)
        ss_tot = torch.sum((y_test_device - torch.mean(y_test_device)) ** 2)
        r2 = (1 - ss_res / ss_tot).item()
        n = x_test_device.shape[0]
        p = x_test_device.shape[1]
        r2_adj = (1 - (1 - r2) * (n - 1) / (n - p - 1))

        # Accuracy
        epsilon = 0.1
        accuracy = (torch.abs(pred - y_test_device) < epsilon)
        accuracy = accuracy.float().mean().item() * 100.0

    # Convert back to original target scale
    target_std = float(y_scaler.scale_[0])

    mse_original = mse_normalized * target_std ** 2
    rmse_original = rmse_normalized * target_std
    
    return (
        mse_original,
        rmse_original,
        r2_adj,
        accuracy,
        elapsed
    )

# ============================================================
# RUN
# ============================================================

print()
print("Device:", DEVICE)
print ("Epochs:", EPOCHS)
print("Model size:", MODEL_SIZE)
print("Architecture:", WIDTHS)
print("Number of layers:", len(WIDTHS) - 1)
print("Grid size:", GRID_SIZE)
print("Spline order:", SPLINE_ORDER)
print("N_spline:", GRID_SIZE + SPLINE_ORDER)

models = build_models(SEED, WIDTHS, GRID_SIZE, SPLINE_ORDER)

results = []

for name, model in models:
    set_seed(SEED)
    
    print()
    print("=" * 60)
    print(name)
    print("=" * 60)

    params = count_parameters(model)

    print(
        "Total parameters:",
        params
    )

    mse, rmse, r2_adj, accuracy, elapsed = train_model(model)

    print()
    print("Results:")

    print(
        f"MSE           = {mse:.6f}"
    )

    print(
        f"RMSE          = {rmse:.6f}"
    )

    print(
        f"Adjusted R²   = {r2_adj:.6f}"
    )

    print(
        f"Accuracy      = {accuracy:.2f}%"
    )

    print(
        f"Time          = {elapsed:.2f} sec"
    )

    results.append(
        (
            name,
            params,
            mse,
            rmse,
            r2_adj,
            accuracy,
            elapsed,
        )
    )


# ============================================================
# FINAL TABLE
# ============================================================

print()
print()

print(
    "=" * 80
)

print(
    "FINAL RESULTS"
)

print(
    "=" * 80
)

print(
    f"{'Model':30s}"
    f"{'Params':>10s}"
    f"{'MSE':>15s}"
    f"{'RMSE':>15s}"
    f"{'Adj R²':>12s}"
    f"{'Accuracy':>12s}"
    f"{'Time':>12s}"
)

print(
    "-" * 80
)


for (
    name,
    params,
    mse,
    rmse,
    r2_adj,
    accuracy,
    elapsed,
) in results:

    print(
        f"{name:30s}"
        f"{params:10d}"
        f"{mse:15.6f}"
        f"{rmse:15.6f}"
        f"{r2_adj:12.6f}"
        f"{accuracy:11.2f}%"
        f"{elapsed:12.2f}s"
    )