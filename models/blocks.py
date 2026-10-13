import torch.nn as nn

from common.layers import SaKANLayer, KALayer

class SaKANBlock(nn.Module):
    """
    Stack of SaKANLayer modules (shared-spline KAN).
    `grad_free=True` detaches the spline basis from autograd.
    """

    def __init__(self, widths, grid_size, spline_order, grad_free=True):
        super().__init__()

        self.layers = nn.ModuleList()

        for i in range(len(widths) - 1):
            self.layers.append(
                SaKANLayer(
                    widths[i],
                    widths[i + 1],
                    grid_size=grid_size,
                    spline_order=spline_order,
                    grad_free=grad_free,
                )
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class SaKANPixelBlock(nn.Module):
    """
    Applies a KA layer independently to every spatial location.

    [B,C,H,W] -> [B,H,W,C] -> KA -> [B,C_out,H,W]

    This gives the segmentation network a simple KA feature-mixing block
    without flattening the whole image.
    """

    def __init__(self, widths, grid_size, spline_order, grad_free=True):
        super().__init__()
        layers = []

        for i in range(len(widths) - 1):
            layers.append(
                SaKANLayer(
                    widths[i],
                    widths[i + 1],
                    grid_size=grid_size,
                    spline_order=spline_order,
                    grad_free=grad_free
                )
            )
            if i < len(widths) - 2:
                layers.append(nn.SiLU())

        self.ka = nn.Sequential(*layers)
            
    def forward(self, x):
        b, c, h, w = x.shape

        x = x.permute(0, 2, 3, 1).reshape(-1, c)
        x = self.ka(x)
        x = x.reshape(b, h, w, -1)

        return x.permute(0, 3, 1, 2)

class KANBlock(nn.Module):
    """
    Reproduces the original KAN formulation with per-edge splines.
    """
    def __init__(
        self,
        widths,
        grid_size = 5,
        splie_order = 3
    ):

        super().__init__()

        self.layers = nn.ModuleList()

        for i in range(len(widths) - 1):

            self.layers.append(
                KALayer(
                    widths[i],
                    widths[i + 1],
                    grid_size=grid_size,
                    spline_order=splie_order
                )
            )


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x