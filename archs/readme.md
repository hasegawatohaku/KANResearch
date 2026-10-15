# Architectures

This folder contains the model architectures of every task and the
`build_models` function that the experiment script imports. Each file
corresponds to one task. Building blocks live in `models/`, layers in
`common/layers/`.

---

## `regression.py` — California Housing regression

- **`MLP`** — plain feed-forward baseline.
  `Linear + SiLU` stack, last layer linear.

- **`build_models(widths, grid_size, spline_order)`** — constructs the
  four models at once and returns them as `(name, model)` pairs:
  - `SAKAN + GRAD-FREE` → `SaKANBlock(..., grad_free=True)`
  - `SAKAN + FULL GRADIENT` → `SaKANBlock(..., grad_free=False)`
  - `ORIGINAL KAN` → `KANBlock(...)`: one spline per input–output edge;
    square hidden layers are initialized with identity + noise
    (`common.init.init_param`)
  - `MLP` → `MLP(widths)`

---

## `classification.py` — CIFAR-100 classification

All four models take the same arguments and return logits. The feature
map of the extractor is flattened; the head has
`channels[-1] × (image_size / 2**len(pools))²` inputs.

- **`ALLUKAN_GFS`** — KAonv extractor + KA classifier head, Grad-Free
  Spline in all layers.
- **`ALLUKAN_FULL_GRAD`** — same, but with full spline gradients in all
  layers.
- **`ALLUKAN_CNN_CONV`** — CNN extractor (`Conv2d + SiLU`) + KA classifier
  head with Grad-Free Spline.
- **`CNN`** — pure CNN baseline: CNN extractor + CNN classifier
  (`Linear + SiLU`).

- **`build_models(in_channels, model, grid_size, spline_order, image_size)`**
  — reads the config dict `model` (`channels`, `ka_widths`, `pools`, `fc`),
  constructs the four models at once and returns them as
  `(name, model)` pairs.

---

## `segmentation.py` — BUSI binary segmentation

All five models are U-Net-shaped: encoder → bottleneck → decoder → 1×1
output.

- Encoder: one 3×3 block per entry of `channels`; max pooling after the
  blocks listed in `pools`. The output of every block (before pooling) is
  kept as a skip connection.
- Bottleneck: pixel-wise KA layers with widths
  `[channels[-1]] + ka_widths`. With `ka_widths = []` it is empty
  (identity).
- Decoder: for every pooled block, in reverse order, bilinear upsampling
  to the size of its skip connection, concatenation with the skip and one
  3×3 block.

Models:

- **`ALLUKAN_GFS`** — KAonv encoder + KA bottleneck + KAonv decoder +
  1×1 KAonv output, Grad-Free Spline in all layers.
- **`ALLUKAN_FULL_GRAD`** — same, but with full spline gradients in all
  KA / KAonv layers.
- **`ALLUKAN_CNN_CONV`** — CNN encoder + KA bottleneck + CNN decoder +
  1×1 `Conv2d` output. With `ka_widths = []` it contains no KA layer and
  has the same architecture as `CNN`.
- **`ALLUKAN_KANELE`** — `ALLUKAN_GFS` prepared for LUT-based inference
  (KANELÉ) [6]. Same architecture, parameter layout and initialization
  scheme as `ALLUKAN_GFS`; built from the same random state, it receives
  exactly the same initial weights. Every KA / KAonv layer additionally:
  - quantizes its inputs to `bits` bits — `input_range` for the first
    encoder layer, `qrange` for all other layers;
  - holds a connection mask `[O, P]` used for pruning;
  - computes its output as an integer sum of table entries with
    `table_frac_bits` fractional bits and quantizes the sum to `qrange`
    (`output_range` for the 1×1 output layer).

  The experiment script prunes connections after every epoch
  (`common/pruning.py`) and, after training, replaces every KA / KAonv
  layer with integer lookup tables (`train/testlut/lut_inf_model.py`).
- **`CNN`** — pure CNN U-Net baseline (`Conv2d + SiLU` blocks, 1×1
  `Conv2d` output).

- **`build_models(...)`** — returns the five models as
  `(name, builder, is_kanele)` triples:
  - `builder` — a function without arguments that constructs the model.
    Models are constructed lazily (one at a time) because the
    Full-Gradient variant needs much more memory.
  - `is_kanele` — `True` only for `ALLUKAN_KANELE`. The experiment uses it
    to enable pruning during training and LUT inference after training.
