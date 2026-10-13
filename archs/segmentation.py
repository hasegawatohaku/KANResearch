import torch.nn as nn

from common.layers import SaKANConv2dLayer

from models.blocks import SaKANPixelBlock
from models.decoders import KADecoder, CNNDecoder
from models.extractors import KAConvFeatureExtractor, CNNFeatureExtractor

class ALLUKAN_GFS(nn.Module):
    """ALL U-KAN-style U-Net with Grad-Free Spline."""

    def __init__(self, 
        ka_widths, 
        in_channels, 
        out_channels, 
        channels, 
        pools, 
        grid_size, 
        spline_order
    ):
        super().__init__()

        self.encoder = KAConvFeatureExtractor(
            in_channels,
            channels,
            pools,
            grid_size=grid_size,
            spline_order=spline_order,
            grad_free=True
        )

        self.bottleneck = SaKANPixelBlock(
            [channels[-1]] + ka_widths,
            grid_size=grid_size,
            spline_order=spline_order,
            grad_free=True
        )

        self.decoder = KADecoder(
            channels,
            pools,
            grid_size=grid_size,
            spline_order=spline_order,
            grad_free=True
        )

        self.output = SaKANConv2dLayer(
            channels[0],
            out_channels,
            kernel_size=1,
            padding=0,
            grid_size=grid_size,
            spline_order=spline_order,
            grad_free=True
        )

    def forward(self, x):
        x, skips = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, skips)
        return self.output(x)

class ALLUKAN_FULL_GRAD(nn.Module):
    """ALL U-KAN-style U-Net without Grad-Free Spline."""

    def __init__(self, 
        ka_widths, 
        in_channels, 
        out_channels, 
        channels, 
        pools, 
        grid_size, 
        spline_order
    ):
        super().__init__()

        self.encoder = KAConvFeatureExtractor(
            in_channels,
            channels,
            pools,
            grid_size=grid_size,
            spline_order=spline_order,
            grad_free=True
        )

        self.bottleneck = SaKANPixelBlock(
            [channels[-1]] + ka_widths,
            grid_size=grid_size,
            spline_order=spline_order,
            grad_free=False
        )

        self.decoder = KADecoder(
            channels,
            pools,
            grid_size=grid_size,
            spline_order=spline_order,
            grad_free=True
        )

        self.output = SaKANConv2dLayer(
            channels[0],
            out_channels,
            kernel_size=1,
            padding=0,
            grid_size=grid_size,
            spline_order=spline_order,
            grad_free=False
        )

    def forward(self, x):
        x, skips = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, skips)
        return self.output(x)

class ALLUKAN_CNN_CONV(nn.Module):
    """
    Modified ALL U-KAN + GFS.

    Convolutional parts are replaced with ordinary CNN convolutions,
    while the bottleneck remains a Grad-Free KA block.
    """

    def __init__(self,
        ka_widths, 
        in_channels, 
        out_channels, 
        channels, 
        pools,
        grid_size, 
        spline_order
    ):
        super().__init__()

        self.encoder = CNNFeatureExtractor(in_channels, channels, pools)

        self.bottleneck = SaKANPixelBlock(
            [channels[-1]] + ka_widths,
            grid_size=grid_size,
            spline_order=spline_order,
            grad_free=True
        )

        self.decoder = CNNDecoder(channels, pools)

        self.output = nn.Conv2d(
            channels[0],
            out_channels,
            kernel_size=1
        )

    def forward(self, x):
        x, skips = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, skips)
        return self.output(x)

class CNN(nn.Module):
    """Simple conventional U-Net-style CNN baseline."""

    def __init__(self, in_channels, out_channels, channels, pools):
        super().__init__()

        self.encoder = CNNFeatureExtractor(in_channels, channels, pools)
        self.decoder = CNNDecoder(channels, pools)

        self.output = nn.Conv2d(
            channels[0],
            out_channels,
            kernel_size=1
        )

    def forward(self, x):
        x, skips = self.encoder(x)
        x = self.decoder(x, skips)
        return self.output(x)

# Models are constructed one at a time in the RUN section.
# This is important for the Full-Gradient model because its spline
# computation can require substantially more memory.
def build_models(widths, 
                 in_channels, 
                 out_channels, 
                 encoder_channels, 
                 pools, 
                 grid_size, 
                 spline_order
):
    return [
        ("ALL U-KAN + GFS", lambda: ALLUKAN_GFS(widths,
                                                in_channels, 
                                                out_channels, 
                                                encoder_channels, 
                                                pools, 
                                                grid_size, 
                                                spline_order)),
        ("ALL U-KAN + FULL GRAD", lambda: ALLUKAN_FULL_GRAD(widths,
                                                            in_channels, 
                                                            out_channels, 
                                                            encoder_channels, 
                                                            pools, 
                                                            grid_size, 
                                                            spline_order)),
        ("ALL U-KAN + CNN CONV", lambda: ALLUKAN_CNN_CONV(widths,
                                                          in_channels, 
                                                          out_channels, 
                                                          encoder_channels, 
                                                          pools, 
                                                          grid_size, 
                                                          spline_order)),
        ("CNN", lambda: CNN(in_channels, 
                            out_channels, 
                            encoder_channels, 
                            pools)),
    ]