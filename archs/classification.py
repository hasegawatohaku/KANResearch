import torch.nn as nn

from models.classifiers import KAClassifier, CNNClassifier
from models.extractors import KAConvFeatureExtractor, CNNFeatureExtractor 

from common.utils import set_seed

class ALLUKAN_GFS(nn.Module):
    """ALL U-KAN: KAonv + KA blocks + Grad-Free Spline"""

    def __init__(self, 
                 in_channels, 
                 channels, 
                 ka_widths, 
                 pools, 
                 grid_size, 
                 spline_order,
                 image_size
    ):
        super().__init__()

        self.features = KAConvFeatureExtractor(
            in_channels,
            channels,
            pools,
            grid_size=grid_size,
            spline_order=spline_order,
            grad_free=True
        )

        self.classifier = KAClassifier(
            channels,
            ka_widths,
            pools,
            grid_size=grid_size,
            spline_order=spline_order,
            image_size=image_size,
            grad_free=True
        )
    
    def forward(self, x):
        x, _ = self.features(x)
        return self.classifier(x)

class ALLUKAN_FULL_GRAD(nn.Module):
    """ALL U-KAN: KAonv + KA blocks + Full Spline Gradients"""

    def __init__(self, 
                 in_channels, 
                 channels, 
                 ka_widths, 
                 pools, 
                 grid_size, 
                 spline_order,
                 image_size
    ):
        super().__init__()

        self.features = KAConvFeatureExtractor(
            in_channels,
            channels,
            pools,
            grid_size=grid_size,
            spline_order=spline_order,
            grad_free=False,
        )

        self.classifier = KAClassifier(
            channels,
            ka_widths,
            pools,
            grid_size=grid_size,
            spline_order=spline_order,
            image_size=image_size,
            grad_free=False,
        )

    def forward(self, x):
        x, _ = self.features(x)
        return self.classifier(x)

class ALLUKAN_CNN_CONV(nn.Module):
    """
    Hybrid model: ordinary CNN feature extractor (Conv2d + SiLU) and a
    KA classifier head with Grad-Free Spline.
    """

    def __init__(self, 
                 in_channels, 
                 channels, 
                 ka_widths, 
                 pools, 
                 grid_size, 
                 spline_order,
                 image_size
    ):
        super().__init__()

        self.features = CNNFeatureExtractor(in_channels, channels, pools)

        self.classifier = KAClassifier(
            channels,
            ka_widths,
            pools,
            grid_size=grid_size,
            spline_order=spline_order,
            image_size=image_size,
            grad_free=True,
        )

    def forward(self, x):
        x, _ = self.features(x)
        return self.classifier(x)

class CNN(nn.Module):
    """Pure conventional CNN: Conv2d + SiLU + pooling + Linear [5]"""

    def __init__(self, 
                 in_channels, 
                 channels, 
                 fc_width, 
                 pools, 
                 image_size
    ):
        super().__init__()

        self.features = CNNFeatureExtractor(in_channels, channels, pools)

        self.classifier = CNNClassifier(
            channels,
            fc_width,
            pools,
            image_size
        )

    def forward(self, x):
        x, _ = self.features(x)
        return self.classifier(x)
    
# Builds the four classification models for the KAonv vs CNN comparison.
def build_models(seed,
                 in_channels, 
                 model, 
                 grid_size, 
                 spline_order, 
                 image_size
):
    model_specs = [
        (
            "ALL U-KAN + GFS",
            lambda: ALLUKAN_GFS(
                in_channels,
                model["channels"],
                model["ka_widths"],
                model["pools"],
                grid_size,
                spline_order,
                image_size
            )
        ),
        (
            "ALL U-KAN + FULL GRAD",
            lambda: ALLUKAN_FULL_GRAD(
                in_channels,
                model["channels"],
                model["ka_widths"],
                model["pools"],
                grid_size,
                spline_order,
                image_size
            )
        ),
        (
            "ALL U-KAN + CNN CONV",
            lambda: ALLUKAN_CNN_CONV(
                in_channels,
                model["channels"],
                model["ka_widths"],
                model["pools"],
                grid_size,
                spline_order,
                image_size
            )
        ),
        (
            "CNN",
            lambda: CNN(
                in_channels,
                model["channels"],
                model["fc"],
                model["pools"],
                image_size
            )
        )
    ]

    models = []

    for name, builder in model_specs:
        set_seed(seed)
        models.append(
            (
                name,
                builder(),
            )
        )

    return models