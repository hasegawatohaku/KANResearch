import torch.nn as nn

from common.layers.kan import KALayer
from common.layers.sakan import SaKANLayer
from common.layers.kanele import KANELEKALayer

class KANBlock(nn.Module):
    """
    Reproduces the original KAN formulation with per-edge splines.
    """
    def __init__(
        self,
        widths,
        grid_size = 5,
        spline_order = 3
    ):
        super().__init__()

        self.blocks = nn.ModuleList()

        for i in range(len(widths) - 1):

            self.blocks.append(
                KALayer(
                    widths[i],
                    widths[i + 1],
                    grid_size=grid_size,
                    spline_order=spline_order
                )
            )


    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class SaKANBlock(KANBlock):
    """
    Stack of SaKANLayer modules (shared-spline KAN).
    `grad_free=True` detaches the spline basis from autograd.
    """

    def __init__(
        self, 
        widths, 
        grid_size=5, 
        spline_order=3, 
        grad_free=True
    ):
        nn.Module.__init__(self)

        self.blocks = nn.ModuleList()

        for i in range(len(widths) - 1):
            self.blocks.append(
                SaKANLayer(
                    widths[i],
                    widths[i + 1],
                    grid_size=grid_size,
                    spline_order=spline_order,
                    grad_free=grad_free,
                )
            )

class SaKANPixelBlock(nn.Module):
    """
    Applies a KA layer independently to every spatial location.

    [B,C,H,W] -> [B,H,W,C] -> KA -> [B,C_out,H,W]

    This gives the segmentation network a simple KA feature-mixing block
    without flattening the whole image.
    """

    def __init__(self, widths, grid_size, spline_order, grad_free=True):
        super().__init__()

        self.blocks = nn.ModuleList()

        for i in range(len(widths) - 1):
            self.blocks.append(
                SaKANLayer(
                    widths[i],
                    widths[i + 1],
                    grid_size=grid_size,
                    spline_order=spline_order,
                    grad_free=grad_free
                )
            )
            if i < len(widths) - 2:
                self.blocks.append(nn.SiLU())
            
    def forward(self, x):
        b, c, h, w = x.shape

        x = x.permute(0, 2, 3, 1).reshape(-1, c)

        for block in self.blocks:
            x = block(x)

        x = x.reshape(b, h, w, -1)

        return x.permute(0, 3, 1, 2)

class KANELEPixelBlock(SaKANPixelBlock):
    """
    SaKANPixelBlock built from KANELEKALayer (same data flow). The SiLU
    between two KA layers is folded into the tables of the second layer
    (pre_silu=True).

    [B,C,H,W] -> [B,H,W,C] -> KA -> [B,C_out,H,W]
    """

    def __init__(
        self,
        widths,
        grid_size,
        spline_order,
        bits,
        qrange,
        table_frac_bits
    ):
        nn.Module.__init__(self)

        self.blocks = nn.ModuleList()

        for i in range(len(widths) - 1):
            self.blocks.append(
                KANELEKALayer(
                    widths[i],
                    widths[i + 1],
                    grid_size=grid_size,
                    spline_order=spline_order,
                    bits=bits,
                    in_range=qrange,
                    out_range=qrange,
                    table_frac_bits=table_frac_bits,
                    pre_silu=(i > 0)
                )
            )