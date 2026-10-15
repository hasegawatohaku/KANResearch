# Configuration files

## ATTENTION!

The values in the config files have been changed many times during
testing and debugging. They are therefore not necessarily well-tuned or
recommended settings, and results obtained with them should not be
compared directly with the results reported in the papers.

This document only describes which parameters each experiment uses and
what they mean, not which values to choose.

---

## `configs/cfgany.py`

Settings shared by all experiments.

| Parameter | Meaning |
|---|---|
| `SEED` | Global random seed. |
| `DEVICE` | `"cuda"` if available, otherwise `"cpu"`. |
| `TIME` | Timestamp string, used to name log files. |

---

## `configs/cfgregression.py`

Configures the California Housing regression experiment.

| Parameter | Meaning |
|---|---|
| `TASK` | Task name. |
| `GRID_SIZE` | Number of B-spline grid intervals. |
| `SPLINE_ORDER` | B-spline degree (`3` = cubic splines; the name follows the KAN convention). The number of basis functions is `GRID_SIZE + SPLINE_ORDER`. |
| `BATCH_SIZE` | Mini-batch size. |
| `EPOCHS` | Number of training epochs. |
| `LR` | Adam learning rate. |
| `MODEL_SIZE` | Key into `ARCHITECTURES`: `"tiny"`, `"small"`, `"medium"`. |
| `ARCHITECTURES` | Dict mapping size → layer widths list. |
| `WIDTHS` | Active layer widths list. |

The width list always starts with `8` (input features) and ends with `1`
(regression output).

---

## `configs/cfgclassification.py`

Configures the CIFAR-100 classification experiment.

| Parameter | Meaning |
|---|---|
| `TASK` | Task name. |
| `GRID_SIZE`, `SPLINE_ORDER` | B-spline parameters. |
| `BATCH_SIZE` | Mini-batch size. |
| `EPOCHS` | Number of training epochs. |
| `LR` | Adam learning rate. |
| `IN_CHANNELS` | Number of input image channels. |
| `IMAGE_SIZE` | Spatial resolution (32). |
| `NUM_CLASSES` | Number of output classes (100). |
| `MODEL_SIZE` | Key into `ARCHITECTURES`. |
| `ARCHITECTURES` | Dict of architecture presets. |
| `MODEL_CFG` | Active preset. |

Each architecture preset contains:

- `channels` — list of output channels for the feature extractor.
- `pools` — list of block indices after which max pooling is applied.
- `fc` — output widths of the CNN classifier head layers (the last one is
  `NUM_CLASSES`).
- `ka_widths` — output widths of the KA classifier head layers (the last
  one is `NUM_CLASSES`).

Presets are named `[conv blocks + hidden head layers]`; the output layer
of the head is not counted:

- `custom[6+1]` — 6 conv blocks, 1 hidden head layer.
- `tiny[2+1]` — 2 conv blocks, 1 hidden head layer.
- `small[3+2]` — 3 conv blocks, 2 hidden head layers.

---

## `configs/cfgsegmentation.py`

Configures the BUSI binary-segmentation experiment.

| Parameter | Meaning |
|---|---|
| `TASK` | Task name. |
| `GRID_SIZE`, `SPLINE_ORDER` | B-spline parameters. |
| `BATCH_SIZE` | Mini-batch size. |
| `EPOCHS` | Number of training epochs. |
| `LR` | Adam learning rate. |
| `IN_CHANNELS` | Input channels (1, grayscale ultrasound). |
| `IMAGE_SIZE` | Spatial resolution. |
| `OUT_CHANNELS` | Output channels (1, binary mask). |
| `DATASET_ROOT` | Path to `Dataset_BUSI_with_GT`. |
| `ENCODER_CHANNELS` | Output channels of each encoder block. |
| `POOLS` | Indices of encoder blocks after which max pooling is applied. |
| `KA_WIDTHS` | Output widths of the KA bottleneck layers. |

Notes:

- `POOLS` also determines the decoder: it lists which encoder blocks
  produce skip connections that the decoder must consume.
- The bottleneck widths are `[ENCODER_CHANNELS[-1]] + KA_WIDTHS`.
  `KA_WIDTHS = []` means the bottleneck is empty (identity): no KA layer
  is applied, and `ALL U-KAN + CNN CONV` then has the same architecture
  as `CNN`. One pixel-wise KA layer with the same number of channels is
  `KA_WIDTHS = [ENCODER_CHANNELS[-1]]`.

---

## `configs/cfglut.py`

Configures the KANELÉ variant of the segmentation experiment
(`ALL U-KAN + KANELE`). The architecture parameters of
`configs/cfgsegmentation.py` are shared with `ALL U-KAN + GFS`.

| Parameter | Meaning |
|---|---|
| `KANELE_QAT_NAME`, `KANELE_LUT_NAME` | Names of the trained (QAT) model and of its LUT model in the results. |
| `KANELE_BITS` | Activation precision in bits = LUT address width. Every table has `2**KANELE_BITS` entries; the table size doubles with every bit. |
| `KANELE_QMIN`, `KANELE_QMAX` | Quantization range of the hidden activations (inputs and outputs of the hidden KA / KAonv layers). The range is shifted by at most half a step so that 0 is a quantization level. Values outside the range are clipped and receive no gradient. |
| `KANELE_INPUT_RANGE` | Quantization range of the input image (first encoder layer). Must match the input scaling: `(0.0, 1.0)` for pixels divided by 255, which with `KANELE_BITS = 8` maps pixel `k / 255` exactly to LUT address `k`. |
| `KANELE_OUTPUT_RANGE` | Quantization range of the logits (1×1 output layer). |
| `KANELE_TABLE_FRAC_BITS` | Fractional bits `F` of the integer table entries (unit = output quantization step / `2**F`). `0` is the format of the KANELÉ code; every extra bit widens every table entry by one bit. |
| `KANELE_TARGET_SPARSITY` | Fraction of connections pruned in every KA / KAonv layer by the last epoch. |
| `KANELE_PRUNING_START_EPOCH` | First epoch after which connections are pruned. |
| `KANELE_CALIBRATION_IMAGES` | Number of fixed training images used to score connections for pruning. |
| `KANELE_CHUNK_ELEMENTS` | Maximum number of table values gathered at once during LUT inference. Limits memory only; the result does not change. |
| `KANELE_EXPORT_LUT` | Save the integer lookup tables to `kanele_segmentation_lut <TIME>.json`. |

Notes:

- **FPGA cost.** `KANELE_BITS` sets the size of every table,
  `KANELE_TARGET_SPARSITY` the number of tables and
  `KANELE_TABLE_FRAC_BITS` the width of every entry. The quantization
  ranges affect accuracy only.
- **Pruning.** After every epoch from `KANELE_PRUNING_START_EPOCH`, the
  connections with the smallest RMS contribution on the calibration
  images are removed. The pruned fraction grows linearly and reaches
  `KANELE_TARGET_SPARSITY` at epoch `EPOCHS`, so a different `EPOCHS`
  changes the pruning rate; compare runs at equal `EPOCHS`. The last
  connection of an output is never removed, and a removed connection is
  never restored.
- **Spline domain.** The spline grid covers [−3, 3] and the spline term
  is zero outside ±6.6 (`GRID_SIZE = 5`, `SPLINE_ORDER = 3`). With a wider
  quantization range, such as ±16, the levels outside ±6.6 pass only
  through the SiLU residual.
- **Output range.** The mask depends only on the sign of the logit, so
  `KANELE_OUTPUT_RANGE` does not change the predicted mask for given
  weights, nor the number or size of the tables. It does change training:
  logits are clipped to the range, which bounds the BCE loss from below
  and stops the gradient outside the range. For `(-3, 3)` the predicted
  probability is limited to [0.047, 0.953] and `Val Loss` is not
  comparable with the other models; a wide range such as `(-16, 16)`
  removes this bound.
- **SaKAN with `KANELE_TABLE_FRAC_BITS = 0`.** The SaKAN layers average
  the spline term over the inputs, so the share of one connection is
  `spline / P`. In wide KAonv layers (`P` up to `9 × (in + skip channels)`)
  it is smaller than one output step and is rounded to zero; `F > 0`
  preserves it.
- **Integer range.** During training, the integer table sums are exact in
  float32 only while `P × max|entry| < 2**24`, where `max|entry|` is
  about `2**(KANELE_BITS − 1 + KANELE_TABLE_FRAC_BITS)` for a symmetric
  range. Otherwise the model raises `ValueError`.