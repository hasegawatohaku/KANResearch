import torch
import torch.nn as nn
import torch.nn.functional as F

from common.layers.sakan import SaKANConv2dLayer
from common.layers.kanele import KANELEConv2dLayer

class KADecoder(nn.Module):
    """
    U-Net decoder built from SaKANConv2dLayer blocks.
    Upsamples the current feature map, concatenates the matching skip, then applies a KA convolution.
    """
    def __init__(self, channels, pools, grid_size, spline_order, grad_free=True):
        super().__init__()

        in_channels = channels[-1]
        
        self.pools = pools
        self.blocks = nn.ModuleList()

        for pool_idx in reversed(self.pools):
            skip_channels = channels[pool_idx]

            self.blocks.append(
                SaKANConv2dLayer(
                    in_channels + skip_channels,
                    skip_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    grad_free=grad_free
                )
            )
            in_channels = skip_channels

    def forward(self, x, skips):
        for block, pool_idx in zip(self.blocks, reversed(self.pools)):
            skip = skips[pool_idx]

            x = F.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

            x = torch.cat([x, skip], dim=1)

            x = block(x)

        return x

class CNNDecoder(KADecoder):
    """
    U-Net decoder built from ordinary Conv2d + SiLU blocks.
    """
    def __init__(self, channels, pools):
        nn.Module.__init__(self)

        in_channels = channels[-1]

        self.blocks = nn.ModuleList()
        self.pools = pools

        for pool_idx in reversed(self.pools):
            skip_channels = channels[pool_idx]
            
            self.blocks.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels + skip_channels,
                        skip_channels,
                        kernel_size=3,
                        padding=1
                    ),
                    nn.SiLU()
                )
            )
            in_channels = skip_channels

class KANELEDecoder(KADecoder):
    """KADecoder with KANELEConv2d blocks; forward() is inherited unchanged."""

    def __init__(
        self,
        channels,
        pools,
        grid_size,
        spline_order,
        bits,
        qrange,
        table_frac_bits
    ):
        nn.Module.__init__(self)

        in_channels = channels[-1]

        self.pools = pools
        self.blocks = nn.ModuleList()

        for pool_idx in reversed(self.pools):
            skip_channels = channels[pool_idx]

            self.blocks.append(
                KANELEConv2dLayer(
                    in_channels + skip_channels,
                    skip_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    bits=bits,
                    in_range=qrange,
                    out_range=qrange,
                    table_frac_bits=table_frac_bits
                )
            )
            in_channels = skip_channels