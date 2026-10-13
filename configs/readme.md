# Configuration files

Every experiment reads its parameters from a dedicated module in
`configs/`. Config files contain **only constants** — no prints, no
dataset loading, no side effects. The experiment scripts import the
constants they need.

---

## `configs/cfgany.py`

Generic / example configuration used as a template.

| Parameter | Meaning |
|---|---|
| `SEED` | Global random seed. |
| `DEVICE` | `"cuda"` if available, otherwise `"cpu"`. |
| `TIME` | Timestamp string, used to name log files. |

Add task-specific parameters here as needed.

---

## `configs/cfgregression.py`

Configures the California Housing regression experiment.

| Parameter | Meaning |
|---|---|
| `GRID_SIZE` | Number of B-spline grid intervals. |
| `SPLINE_ORDER` | B-spline order (degree + 1). |
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
- `fc` — widths of the CNN classifier head (including `NUM_CLASSES`).
- `ka_widths` — widths of the KA classifier head (including `NUM_CLASSES`).

Presets:

- `custom[6+1]` — 6 conv blocks + 1 head layer.
- `tiny[2+1]` — 2 conv blocks + 1 head layer.
- `small[3+2]` — 3 conv blocks + 2 head layers.

---

## `configs/kan_conv_vs_cnn_segmentation.py`

Configures the BUSI binary-segmentation experiment.

| Parameter | Meaning |
|---|---|
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
| `KA_WIDTHS` | Widths of the KA bottleneck. |

Notes:

- `POOLS` also determines the decoder: it lists which encoder blocks
  produce skip connections that the decoder must consume.
- `KA_WIDTHS = []` means the bottleneck is a single `SaKANLayer` applied
  pixel-wise with the same number of channels.