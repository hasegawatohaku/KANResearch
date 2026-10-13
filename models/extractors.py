import torch.nn as nn
import torch.nn.functional as F

from common.layers import SaKANConv2dLayer

class KAConvFeatureExtractor(nn.Module):
    """
    Feature extractor built from SaKANConv2dLayer blocks.
    Max pooling is inserted after the block indices listed in `pools`.
    Returns a single feature map.
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

class CNNFeatureExtractor(nn.Module):
    """
    Ordinary perceptron-based convolutional blocks.

    Conv2d contains scalar trainable weights. SiLU provides the
    nonlinearity between convolutional layers. MaxPool2d reduces
    spatial resolution
    """

    def __init__(self, in_channels, channels, pools):
        super().__init__()

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

    def forward(self, x):
        skips = []

        for i, block in enumerate(self.blocks):
            x = block(x)
            skips.append(x)

            if i in self.pools:
                x = F.max_pool2d(x, 2)

        return x, skips