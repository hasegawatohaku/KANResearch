# KAN vs MLP / KAN-Conv vs CNN — Comparison Suite

A small research codebase that compares **Kolmogorov–Arnold Networks (KAN)**
against classical **MLPs** and **CNNs** across three tasks:

1. Tabular regression (California Housing) — KAN vs MLP.
2. Image classification (CIFAR-100) — KAonv vs CNN.
3. Binary segmentation (BUSI) — KAonv vs CNN in a U-Net-style setup,
   including a hardware-oriented **KANELÉ** variant [6, 7]:
   quantization-aware training, connection pruning and inference through
   integer lookup tables (LUTs).

The repository is organised as a set of importable packages plus thin
experiment scripts that only wire the pieces together.

---

## Project layout

```
.
├── archs/                # Model architectures and build_models() of each task
│
├── common/               # Shared code: B-spline basis, initialization, loss,
│   │                     # dataset, quantization, pruning, logging
│   └── layers/           # Layers: original KAN, SaKAN (KA / KAonv), KANELÉ
│
├── configs/              # Configuration files (constants only)
│
├── data/                 # Datasets (not tracked by git)
│
├── models/               # Building blocks: KA blocks, feature extractors,
│                         # decoders, classifier heads
│
├── res/                  # Log files produced by experiments (not tracked by git)
│
└── train/                # Experiment scripts: training and evaluation loops
    └── testlut/          # Integer LUT inference and LUT export
```

---

## Environment

- Python 3.10+
- PyTorch (with CUDA if available)
- torchvision
- scikit-learn
- NumPy, Pillow

---

## Datasets

All datasets live under `./data/`:

- `data/cifar100/` — CIFAR-100. Downloaded automatically by
  `torchvision.datasets.CIFAR100` on first run.
- `data/Dataset_BUSI_with_GT/` — BUSI. Must be downloaded manually and
  extracted into `./data/Dataset_BUSI_with_GT/`. Expected subfolders:
  `benign/`, `malignant/`, `normal/`. Only `benign` and `malignant`
  images are used (647 image–mask pairs); `normal` images have no lesion.
- California Housing is fetched from `scikit-learn` directly, no local copy.

---

## Results

Every experiment redirects `stdout` into a timestamped `.log` file inside
`./res/`. The same content is still printed to the terminal.

With `KANELE_EXPORT_LUT = True`, the segmentation experiment additionally
saves the integer lookup tables of the KANELÉ model to
`kanele_segmentation_lut <TIME>.json` in the working directory.

---

## Running the experiments

From the repository root, **run experiments as modules**:

```bash
python -m train.regression_models
python -m train.classifier_models
python -m train.segmentation_models
```

For VS Code debugging, use `"module": "train.<name>"` in `launch.json`
with `"cwd": "${workspaceFolder}"`. See the individual config files for
the full list of tunable parameters.

---

## What is compared

| Experiment | Models | Task | Metric |
|---|---|---|---|
| `regression_models` | SaKAN (GFS / full grad), Original KAN, MLP | Regression | MSE, RMSE, Adj R² |
| `classifier_models` | ALL U-KAN (GFS / full grad / CNN-conv), CNN | Classification | Accuracy |
| `segmentation_models` | ALL U-KAN (GFS / full grad / CNN-conv / KANELÉ: QAT and LUT), CNN | Segmentation | IoU, F1 |

---

## Implementation notes

- **SaKAN averages the spline term.** The KA and KAonv layers follow the
  Shared-activation KAN (SaKAN) of ALL U-KAN [4]: one spline per output,
  shared by all inputs, plus a SiLU linear residual. Unlike Eq. 7 of [4],
  which sums the spline term over the inputs, this implementation
  averages it over the inputs (factor `1/N`). The scaling is deliberate:
  without it, models with KAonv layers train very slowly.
- **Grad-Free Spline (GFS)** detaches the B-spline basis from autograd;
  only the spline coefficients `v` remain trainable in the spline branch.
- **Simplified ALL U-KAN.** The "ALL U-KAN" models are deliberately
  reduced to plain KA / KAonv layers (no BatchNorm, ReLU, LayerNorm or
  tokenized KA blocks), so that the layers themselves are compared.
- **Original KAN** uses one spline per input–output edge on a fixed grid
  (no grid update).
- **Spline domain.** All spline layers use a fixed uniform grid on
  [−3, 3], extended by `SPLINE_ORDER` intervals on each side. Outside the
  extended range (±6.6 for `GRID_SIZE = 5`, `SPLINE_ORDER = 3`) the spline
  term is zero and only the SiLU residual acts.
- **Random seed.** Each experiment script calls `common.utils.set_seed(SEED)`
  once at the start. The models are then built and trained one after
  another, so each model takes its initial weights and its batch order
  from a different part of the random stream.

---

## References
1. Liu Z., Wang Y., Vaidya S., Ruehle F., Halverson J., Soljačić M., Hou T. Y., & Tegmark M. (2024).  
   *KAN: Kolmogorov–Arnold Networks.*  
   arXiv preprint arXiv:2404.19756.  
   https://arxiv.org/abs/2404.19756

2. Bodner A. D., Tepsich A. S., Spolski J. N., & Pourteau S. (2024).  
   *Convolutional Kolmogorov–Arnold Networks.*  
   arXiv preprint arXiv:2406.13155.  
   https://arxiv.org/abs/2406.13155

3. Tepsich A., et al.  
   *Convolutional-KANs* [GitHub repository].  
   https://github.com/AntonioTepsich/Convolutional-KANs

4. Qiu X., Ma X., Liang D., Luo G., Wang W., Wang K., & Li S. (2026).  
   *Fully Kolmogorov–Arnold Deep Model in Medical Image Segmentation.*  
   arXiv preprint arXiv:2602.03156.  
   https://arxiv.org/html/2602.03156v1

5. O’Shea K., & Nash R. (2015).  
   *An Introduction to Convolutional Neural Networks.*  
   arXiv preprint arXiv:1511.08458.  
   https://arxiv.org/pdf/1511.08458

6. Hoang D., Gupta A., & Harris P. (2026).  
   *KANELÉ: Kolmogorov–Arnold Networks for Efficient LUT-based Evaluation.*  
   arXiv preprint arXiv:2512.12850.  
   https://arxiv.org/html/2512.12850

7. Hoang D., Gupta A.  
   *KANELE* [GitHub repository].  
   https://github.com/Duchstf/KANELE
