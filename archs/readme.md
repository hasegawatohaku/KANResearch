# Models

This folder contains all model definitions used in the experiments.
Each file corresponds to one task.

---

## `regression.py` — California Housing regression

- **`MLP`** — plain feed-forward baseline.
  `Linear + SiLU` stack, last layer linear.

- **`build_models(widths, grid_size, spline_order)`** — returns four
  models for comparison:
  - `SAKAN + GRAD-FREE`  → `SaKANBlock(..., grad_free=True)`
  - `SAKAN + FULL GRADIENT` → `SaKANBlock(..., grad_free=False)`
  - `ORIGINAL KAN` → `KANBlock(...)`
  - `MLP` → `MLP(widths)`

---

## `classification.py` — CIFAR-100 classification

All four models take the same arguments and return logits.

- **`ALLUKAN_GFS`** — KAonv extractor + KA classifier, Grad-Free Spline.
- **`ALLUKAN_FULL_GRAD`** — same but with full spline gradients.
- **`ALLUKAN_CNN_CONV`** — CNN extractor + KA classifier (Grad-Free in the head).
- **`CNN`** — pure CNN baseline: CNN extractor + CNN classifier.

- **`build_models(in_channels, model, grid_size, spline_order, image_size)`**
  — reads the config dict `model` (`channels`, `ka_widths`, `pools`, `fc`)
  and returns the four models above as `(name, model)` pairs.

---

## `segmentation.py` — BUSI binary segmentation

All four are U-Net-shaped: encoder → bottleneck → decoder → 1×1 output.

- **`ALLUKAN_GFS`** — KA encoder + KA bottleneck + KA decoder,
  Grad-Free Spline everywhere.
- **`ALLUKAN_FULL_GRAD`** — same, but bottleneck and output use full
  spline gradients.
- **`ALLUKAN_CNN_CONV`** — CNN encoder + KA bottleneck + CNN decoder.
- **`ALLUKAN_KANELE`** — `ALLUKAN_GFS` prepared for LUT-based inference
  (KANELÉ) [6]. Same architecture, parameterization and initialization as
  `ALLUKAN_GFS`. Every KA / KAonv layer additionally:
  - quantizes its inputs to `bits` bits — `input_range` for the first
    encoder layer, `qrange` for all other layers;
  - holds a connection mask `[O, P]` used for pruning;
  - computes its output as an integer sum of table entries with
    `table_frac_bits` fractional bits and quantizes the sum to `qrange`
    (`output_range` for the 1×1 output layer).
- **`CNN`** — pure CNN U-Net baseline.

- **`build_models(...)`** — returns the four models above as
  `(name, model)` pairs. Models are constructed lazily (one at a time)
  because the Full-Gradient variant needs much more memory.
