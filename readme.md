# KAN vs MLP / KAN-Conv vs CNN — Comparison Suite

A small research codebase that compares **Kolmogorov–Arnold Networks (KAN)**
against classical **MLPs** and **CNNs** across three tasks:

1. Tabular regression (California Housing) — KAN vs MLP.
2. Image classification (CIFAR-100) — KAonv vs CNN.
3. Binary segmentation (BUSI) — KAonv vs CNN in a U-Net-style setup.

The repository is organised as a set of importable packages plus thin
experiment scripts that only wire the pieces together.

---

## Project layout

```
.
├── archs/                # Markdown descriptions of each architecture
│
├── common/               # Shared building blocks
│
├── configs/              # Plain-data configuration files
│
├── data/                 # Datasets
│
├── models/               # Model definitions
│
├── res/                  # Log files produced by experiments
│
└── train/                # Training and evaluation loops

```

---

## Environment

- Python 3.10+
- PyTorch (with CUDA if available)
- torchvision
- scikit-learn
- numpy
- Pillow

---

## Datasets

All datasets live under `./data/`:

- `data/cifar100/` — CIFAR-100. Downloaded automatically by
  `torchvision.datasets.CIFAR100` on first run.
- `data/Dataset_BUSI_with_GT/` — BUSI. Must be downloaded manually and extracted into `./data/Dataset_BUSI_with_GT/`. Expected subfolders:
  `benign/`, `malignant/`, `normal/`.
- California Housing is fetched from `scikit-learn` directly, no local copy.

---

## Logs

Every experiment redirects `stdout` into a timestamped `.log` file inside
`./res/`. The same content is still printed to the terminal.

---

## Running the experiments

From the repository root, **run experiments as modules**:

```bash
python -m train.regression_models
python -m train.classifier_models
python -m train.segmentation_models
```

For VS Code debugging, use `"module": "experiments.<name>"` in
`launch.json` with `"cwd": "${workspaceFolder}"`. See the individual
config files for the full list of tunable parameters.

---

## What is compared

| Experiment | Models | Task | Metric |
|---|---|---|---|
| `regression_models` | SaKAN (GFS / full grad), Original KAN, MLP | Regression | MSE, RMSE, Adj R² |
| `classifier_models` | ALL U-KAN (GFS / full grad / CNN-conv), CNN | Classification | Accuracy |
| `segmentation_models` | ALL U-KAN (GFS / full grad / CNN-conv), CNN | Segmentation | IoU, F1 |

---

## Notes

- All randomness is seeded through `common.utils.set_seed`.
- Model construction in each experiment resets the seed so that every
  model starts from the same initialization distribution.
- The Grad-Free Spline variant detaches the B-spline basis from autograd;
  only the spline coefficients `v` remain trainable in the spline branch.

---

## References 
1. Liu, Z., Wang, Y., Vaidya, S., Ruehle, F., Halverson, J., Soljačić, M., Hou, T. Y., & Tegmark, M. (2024).  
   *KAN: Kolmogorov–Arnold Networks.*  
   arXiv preprint arXiv:2404.19756.  
   https://arxiv.org/abs/2404.19756

2. Bodner, A. D., Tepsich, A. S., Spolski, J. N., & Pourteau, S. (2024).  
   *Convolutional Kolmogorov–Arnold Networks.*  
   arXiv preprint arXiv:2406.13155.  
   https://arxiv.org/abs/2406.13155

3. Tepsich, A., et al. (2024).  
   *Convolutional-KANs* [GitHub repository].  
   https://github.com/AntonioTepsich/Convolutional-KANs

4. Qiu, X., Ma, X., Liang, D., Luo, G., Wang, W., Wang, K., & Li, S. (2026).  
   *Fully Kolmogorov–Arnold Deep Model in Medical Image Segmentation.*  
   arXiv preprint arXiv:2602.03156.  
   https://arxiv.org/html/2602.03156v1

5. O’Shea, K., & Nash, R. (2015).  
   *An Introduction to Convolutional Neural Networks.*  
   arXiv preprint arXiv:1511.08458.  
   https://arxiv.org/pdf/1511.08458