import torch
import torch.nn as nn

from common.init import init_param
from common.bspaline import make_knots, bspline_basis

class KALayer(nn.Module):
    """
    Original KAN layer with edge-wise spline coefficients [1]:

        spline[o, i] = sum_k v[o, i, k] * B_k(x_i)

    The SiLU residual branch is the same as in SaKANLayer.
    """
    def __init__(
        self,
        n_in,
        n_out,
        grid_size=5,
        spline_order=3,
    ):

        super().__init__()

        self.n_in = n_in
        self.n_out = n_out

        self.grid_size = grid_size
        self.spline_order = spline_order

        self.n_spline = grid_size + spline_order

        # One spline per input-output edge
        self.v = nn.Parameter(
            torch.empty(
                n_out,
                n_in,
                self.n_spline,
            )
        )


        self.u = nn.Parameter(
            torch.empty(
                n_out,
                n_in,
            )
        )

        # Initialization
        init_param(
            self.u,
            self.v,
            n_in,
            n_out,
        )

        # Fixed knots
        knots = make_knots(
            grid_size,
            spline_order,
        )

        self.register_buffer("knots", knots)

        self.silu = nn.SiLU()


    def forward(self, x):

        basis = bspline_basis(
            x,
            self.knots,
            self.spline_order,
        )

        # Edge-wise spline:
        # basis  = [batch, input, spline]
        # v      = [output, input, spline]
        # ---------------------------------
        # result = [batch, output, input]
        spline = torch.einsum(
            "bik,oik->boi",
            basis,
            self.v,
        )

        # Sum over input dimensions.
        spline = spline.sum(dim=-1)

        # Base SiLU branch
        base = self.silu(x)

        base = torch.einsum(
            "bi,oi->bo",
            base,
            self.u,
        )

        return spline + base