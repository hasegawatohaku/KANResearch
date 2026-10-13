import torch
import torch.nn as nn
import numpy as np


# Initializes the two parameter tensors of a KAN-like layer.
# `u` is the SiLU residual matrix; `v` is the spline coefficient tensor.
# Square hidden layers get an identity + small noise initialization, other layers get Xavier. Spline coefficients start small.
# This keeps deep stacks numerically stable.
def init_param(
    u,
    v,
    n_in,
    n_out,
):
    with torch.no_grad():

        # Hidden square layers
        if n_in == n_out and n_in > 1:
            u.copy_(
                torch.eye(
                    n_in,
                    dtype=u.dtype,
                    device=u.device,
                )
            )
            u.add_(0.01 * torch.randn_like(u))

        # Input/output layers
        else:
            nn.init.xavier_uniform_(u)

        # Spline coefficients
        spline_std = 0.02 / np.sqrt(max(n_in, 1))

        v.normal_(
            mean=0.0,
            std=float(spline_std),
        )