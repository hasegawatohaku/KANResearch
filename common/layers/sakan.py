import torch
import torch.nn as nn
import torch.nn.functional as F

from common.bspaline import make_knots, bspline_basis

class SaKANLayer(nn.Module):
    """
    KA layer based on the Shared-activation KAN (SaKAN) of ALL U-KAN [4]:
    one spline per output, shared by all inputs, plus a SiLU linear residual.

        y_o = (1/N) sum_i sum_k v[o,k] * B_k(x_i) + sum_i u[o,i] * SiLU(x_i)

    Deviation from [4]: Eq. 7 sums the spline term over the N inputs; here
    it is averaged (factor 1/N). Without this scaling, models with KAonv
    layers train very slowly.

    With grad_free=True, the spline basis is detached from the autograd
    graph (Grad-Free Spline); the spline coefficients v remain trainable.
    """

    def __init__(
        self,
        n_in,
        n_out,
        grid_size=5,
        spline_order=3,
        grad_free=True
    ):
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.grad_free = grad_free
        self.spline_order = spline_order

        self.n_spline = grid_size + spline_order

        # Shared spline coefficients:
        # [output, spline_basis]
        self.v = nn.Parameter(
            0.02 * torch.randn(
                n_out,
                self.n_spline,
            )
        )

        # Linear residual coefficients:
        # [output, input]
        self.u = nn.Parameter(
            torch.empty(
                n_out,
                n_in, 
            )
        )

        nn.init.xavier_uniform_(self.u)

        self.register_buffer(
            "knots",
            make_knots(
                grid_size,
                spline_order,
            ),
        )

        self.silu = nn.SiLU()

    def forward(self, x):
        # x: [B, N]

        # B-spline basis:
        # [B,N] -> [B,N,S]
        basis = bspline_basis(
            x,
            self.knots,
            self.spline_order
        )

        # Grad-Free Spline:
        # no gradient is propagated through the basis.
        if self.grad_free:
            basis = basis.detach()
        
        # Shared spline across input dimensions.
        basis = basis.mean(dim=1)          # [B,N,S] -> [B,S]
        spline = basis @ self.v.t()        # [B,O]

        # SiLU residual branch.
        base = self.silu(x)
        base = base @ self.u.t()    # [B,N] x [O,N] -> [B,O]

        return spline + base

class SaKANConv2dLayer(nn.Module):
    """
    KAonv layer of ALL U-KAN [4] (convolution through a KA layer, cf. [2-3]).

    Every local window is unfolded into a patch of P = C * k * k values,
    and the patch is passed through a SaKAN layer: the spline term is
    averaged over the P patch elements (as in SaKANLayer), and the SiLU
    residual has one weight per patch element and output channel.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        grid_size=5,
        spline_order=3,
        grad_free=True
    ):
        super().__init__()

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.grad_free = grad_free
        self.spline_order = spline_order

        patch_dim = in_channels * kernel_size * kernel_size
        n_spline = grid_size + spline_order

        self.v = nn.Parameter(
            0.02 * torch.randn(
                out_channels,
                n_spline
            )
        )

        self.u = nn.Parameter(
            torch.empty(
                out_channels,
                patch_dim
            )
        )

        nn.init.xavier_uniform_(self.u)

        self.register_buffer(
            "knots",
            make_knots(
                grid_size,
                spline_order
            )
        )

        self.silu = nn.SiLU()

    def forward(self, x):
        # [B,C,H,W] -> [B,L,P]
        patches = F.unfold(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding
        ).transpose(1, 2)

        # [B,L,P] -> [B,L,P,S]
        basis = bspline_basis(
            patches,
            self.knots,
            self.spline_order
        )

        if self.grad_free:
            basis = basis.detach()

        # Shared spline across all elements of a local patch.
        basis = basis.mean(dim=2)
        spline = basis @ self.v.t()

        # SiLU residual branch.
        base = self.silu(patches)
        base = base @ self.u.t()

        y = spline + base

        h = (x.shape[2] + 2 * self.padding - self.kernel_size) // self.stride + 1

        w = (x.shape[3] + 2 * self.padding - self.kernel_size) // self.stride + 1

        y = y.transpose(1, 2)

        return y.reshape(
            x.shape[0],
            self.out_channels,
            h,
            w
        )