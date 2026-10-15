import torch
import torch.nn as nn

# ============================================================
# LOSS
# ============================================================
# Combined BCE + soft Dice loss used by ALL U-KAN segmentation.
# BCE provides a stable per-pixel gradient; Dice focuses on the foreground region and helps with class imbalance.
class BCEDiceLoss(nn.Module):
    """
    BCE + soft Dice loss as in the U-KAN code, on which ALL U-KAN builds:

        loss = 0.5 * BCE + (1 - mean Dice over the images of the batch)
    """

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, target):
        bce = self.bce(logits, target)

        probability = torch.sigmoid(logits)

        intersection = (probability * target).sum(dim=(1, 2, 3))

        denominator = probability.sum(dim=(1, 2, 3))+ target.sum(dim=(1, 2, 3))

        dice = (2.0 * intersection + 1e-6) / (denominator + 1e-6)

        dice_loss = 1.0 - dice.mean()

        return 0.5 * bce + dice_loss