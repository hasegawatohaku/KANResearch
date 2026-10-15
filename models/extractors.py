import torch.nn as nn
import torch.nn.functional as F

from common.layers.sakan import SaKANConv2dLayer
from common.layers.kanele import KANELEConv2dLayer

class KAConvFeatureExtractor(nn.Module):
    """
    Feature extractor built from 3x3 SaKANConv2dLayer blocks.
    Max pooling is applied after the blocks whose indices are in `pools`.

    Returns (x, skips): the final feature map and the outputs of all
    blocks before pooling (skip connections for a U-Net decoder).
    """
    def __init__(self, in_channels, channels, pools, grid_size, spline_order, grad_free=True):
        super().__init__()

        self.pools = pools

        self.blocks = nn.ModuleList()

        for out_channels in channels:
            self.blocks.append(
                SaKANConv2dLayer(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    grad_free=grad_free
                )
            )
            in_channels = out_channels

    def forward(self, x):
        skips = []

        for i, block in enumerate(self.blocks):
            x = block(x)
            skips.append(x)

            if i in self.pools:
                x = F.max_pool2d(x, 2)

        return x, skips

class CNNFeatureExtractor(KAConvFeatureExtractor):
    """
    Ordinary perceptron-based convolutional blocks.

    Conv2d contains scalar trainable weights. SiLU provides the
    nonlinearity between convolutional layers. MaxPool2d reduces
    spatial resolution

    forward() function inherits from the KAConvFeatureExtractor.
    """
    def __init__(self, in_channels, channels, pools):
        nn.Module.__init__(self)

        self.pools = pools

        self.blocks = nn.ModuleList()

        for out_channels in channels:
            self.blocks.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        padding=1
                    ),
                    nn.SiLU()
                )
            )
            in_channels = out_channels

class KANELEConvFeatureExtractor(KAConvFeatureExtractor):
    """
    KAConvFeatureExtractor built from KANELEConv2dLayer blocks (quantized,
    prunable, integer tables). The first block quantizes the input image
    with `input_range`; all other blocks use `qrange`.

    forward() is inherited from KAConvFeatureExtractor.
    """

    def __init__(
        self,
        in_channels,
        channels,
        pools,
        grid_size,
        spline_order,
        bits,
        input_range,
        qrange,
        table_frac_bits
    ):
        nn.Module.__init__(self)

        self.pools = pools

        self.blocks = nn.ModuleList()

        for i, out_channels in enumerate(channels):
            self.blocks.append(
                KANELEConv2dLayer(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    bits=bits,
                    # The first layer reads the input image.
                    in_range=(input_range if i == 0 else qrange),
                    out_range=qrange,
                    table_frac_bits=table_frac_bits
                )
            )
            in_channels = out_channels
