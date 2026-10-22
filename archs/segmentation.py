import torch.nn as nn

from common.layers.sakan import SaKANConv2dLayer
from common.layers.kanele import KANELEConv2dLayer

from models.blocks import SaKANPixelBlock, KANELEPixelBlock
from models.decoders import KADecoder, KANELEDecoder, CNNDecoder
from models.extractors import KAConvFeatureExtractor, KANELEConvFeatureExtractor, CNNFeatureExtractor

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

        bottleneck_widths = [channels[-1]] + ka_widths

        self.encoder = KAConvFeatureExtractor(
            in_channels,
            channels,
            pools,
            grid_size=grid_size,
            spline_order=spline_order,
            grad_free=True
        )

        self.bottleneck = SaKANPixelBlock(
            bottleneck_widths,
            grid_size=grid_size,
            spline_order=spline_order,
            grad_free=True
        )

        self.decoder = KADecoder(
            channels,
            pools,
            grid_size=grid_size,
            spline_order=spline_order,
            grad_free=True,
            in_channels=bottleneck_widths[-1]
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

        bottleneck_widths = [channels[-1]] + ka_widths

        self.encoder = KAConvFeatureExtractor(
            in_channels,
            channels,
            pools,
            grid_size=grid_size,
            spline_order=spline_order,
            grad_free=False
        )

        self.bottleneck = SaKANPixelBlock(
            bottleneck_widths,
            grid_size=grid_size,
            spline_order=spline_order,
            grad_free=False
        )

        self.decoder = KADecoder(
            channels,
            pools,
            grid_size=grid_size,
            spline_order=spline_order,
            grad_free=False,
            in_channels=bottleneck_widths[-1]
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
    Modified ALL U-KAN + GFS: the KAonv layers are replaced with ordinary
    CNN convolutions, while the bottleneck remains a Grad-Free KA block.
    With ka_widths = [] the bottleneck is empty and the model has the
    same architecture as CNN.
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

        bottleneck_widths = [channels[-1]] + ka_widths

        self.encoder = CNNFeatureExtractor(in_channels, channels, pools)

        self.bottleneck = SaKANPixelBlock(
            bottleneck_widths,
            grid_size=grid_size,
            spline_order=spline_order,
            grad_free=True
        )

        self.decoder = CNNDecoder(channels, pools)

        self.output = nn.Conv2d(
            channels[0],
            out_channels,
            kernel_size=1,
            in_channels=bottleneck_widths[-1]
        )

    def forward(self, x):
        x, skips = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, skips)
        return self.output(x)

class ALLUKAN_KANELE(nn.Module):
    """
    ALL U-KAN + GFS with KANELÉ quantization-aware training,
    connection pruning and integer LUT inference.

    Architecture, parameter layout and initialization scheme are identical
    to ALLUKAN_GFS; built from the same random state, both models receive
    the same initial weights. Every KA / KAonv layer quantizes its inputs,
    computes its output with integer tables and quantizes the sum.
    """

    def __init__(self,
        ka_widths,
        in_channels,
        out_channels,
        channels,
        pools,
        grid_size,
        spline_order,
        bits=8,
        input_range=(0.0, 1.0),
        qrange=(-16, 16),
        output_range=(-16, 16),
        table_frac_bits=4
    ):
        super().__init__()

        bottleneck_widths = [channels[-1]] + ka_widths

        self.encoder = KANELEConvFeatureExtractor(
            in_channels,
            channels,
            pools,
            grid_size=grid_size,
            spline_order=spline_order,
            bits=bits,
            input_range=input_range,
            qrange=qrange,
            table_frac_bits=table_frac_bits
        )

        self.bottleneck = KANELEPixelBlock(
            bottleneck_widths,
            grid_size=grid_size,
            spline_order=spline_order,
            bits=bits,
            qrange=qrange,
            table_frac_bits=table_frac_bits
        )

        self.decoder = KANELEDecoder(
            channels,
            pools,
            grid_size=grid_size,
            spline_order=spline_order,
            bits=bits,
            qrange=qrange,
            table_frac_bits=table_frac_bits,
            in_channels=bottleneck_widths[-1]
        )

        self.output = KANELEConv2dLayer(
            channels[0],
            out_channels,
            kernel_size=1,
            padding=0,
            grid_size=grid_size,
            spline_order=spline_order,
            bits=bits,
            in_range=qrange,
            out_range=output_range,
            table_frac_bits=table_frac_bits
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
# This is important for the Full-Gradient model because its spline computation can require substantially more memory.
def build_models(widths, 
                 in_channels, 
                 out_channels, 
                 encoder_channels, 
                 pools, 
                 grid_size, 
                 spline_order,
                 bits,
                 input_range,
                 qrange,
                 output_range,
                 table_frac_bits
):
    return [
        ("ALL U-KAN + GFS", lambda: ALLUKAN_GFS(widths,
                                                in_channels, 
                                                out_channels, 
                                                encoder_channels, 
                                                pools, 
                                                grid_size, 
                                                spline_order), False),
        ("ALL U-KAN + FULL GRAD", lambda: ALLUKAN_FULL_GRAD(widths,
                                                            in_channels, 
                                                            out_channels, 
                                                            encoder_channels, 
                                                            pools, 
                                                            grid_size, 
                                                            spline_order), False),
        ("ALL U-KAN + CNN CONV", lambda: ALLUKAN_CNN_CONV(widths,
                                                          in_channels, 
                                                          out_channels, 
                                                          encoder_channels, 
                                                          pools, 
                                                          grid_size, 
                                                          spline_order), False),
        ("ALL U-KAN + KANELE (QAT)", lambda: ALLUKAN_KANELE(widths,
                                                            in_channels,
                                                            out_channels,
                                                            encoder_channels,
                                                            pools,
                                                            grid_size,
                                                            spline_order,
                                                            bits,
                                                            input_range,
                                                            qrange,
                                                            output_range,
                                                            table_frac_bits), True),
        ("CNN", lambda: CNN(in_channels, 
                            out_channels, 
                            encoder_channels, 
                            pools), False)
    ]