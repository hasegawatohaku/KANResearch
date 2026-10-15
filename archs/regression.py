import torch.nn as nn

from models.blocks import SaKANBlock, KANBlock

from common.utils import set_seed

class MLP(nn.Module):
    """
    Standard MLP baseline with SiLU activations.
    The last layer is linear (no activation).
    """
    def __init__(
        self,
        widths,
    ):

        super().__init__()

        self.layers = nn.ModuleList()

        for i in range(len(widths) - 1):

            self.layers.append(
                nn.Linear(
                    widths[i],
                    widths[i + 1],
                )
            )

            if i < len(widths) - 2:
                self.layers.append(nn.SiLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
# Builds all regression models for the comparison.
def build_models(seed, widths, grid_size, spline_order):

    model_specs = [
        (
            "SAKAN + GRAD-FREE",
            lambda: SaKANBlock(
                widths,
                grid_size,
                spline_order,
                grad_free=True,
            )
        ),

        (
            "SAKAN + FULL GRADIENT",
            lambda: SaKANBlock(
                widths,
                grid_size,
                spline_order,
                grad_free=False,
            )
        ),

        (
            "ORIGINAL KAN",
            lambda: KANBlock(
                widths,
                grid_size,
                spline_order
            )
        ),

        (
            "MLP",
            lambda: MLP(
                widths
            )
        )
    ]

    models = []

    for name, builder in model_specs:
        set_seed(seed)
        models.append(
            (
                name,
                builder()
            )
        )

    return models