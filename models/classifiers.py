import torch
import torch.nn as nn

from models.blocks import SaKANBlock

class KAClassifier(nn.Module):
    """
    Flattens the feature map and passes it through a SaKANBlock classifier head.
    """
    def __init__(self, 
                 channels, 
                 ka_widths, 
                 pools,
                 grid_size,
                 spline_order,
                 image_size, 
                 grad_free=True
    ):
        super().__init__()

        final_size = image_size // (2 ** len(pools))

        input_width = channels[-1] * final_size * final_size

        widths = [input_width] + list(ka_widths)

        self.ka = SaKANBlock(
            widths,
            grid_size=grid_size,
            spline_order=spline_order,
            grad_free=grad_free,
        )

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        return self.ka(x)

class CNNClassifier(nn.Module):
    """
    Flattens the feature map and passes it through ordinary Linear layers with SiLU between them.
    """
    def __init__(self, channels, fc_widths, pools, image_size):
        super().__init__()

        final_size = image_size // (2 ** len(pools))

        input_width = channels[-1] * final_size * final_size

        widths = [input_width] + list(fc_widths)

        layers = [nn.Flatten()]

        for i in range(len(widths) - 1):
            layers.append(nn.Linear(widths[i], widths[i + 1]))

            if i < len(widths) - 2:
                layers.append(nn.SiLU())

        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        return self.classifier(x)

