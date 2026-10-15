import torch
import torch.nn as nn

# ============================================================
# LOSS
# ============================================================
# Combined BCE + soft Dice loss used by ALL U-KAN segmentation.
# BCE provides a stable per-pixel gradient; Dice focuses on the foreground region and helps with class imbalance.
class BCEDiceLoss(nn.Module):
    """
    Loss used by ALL U-KAN:

        BCE + Dice loss

    The paper states that the segmentation experiments combine
    Binary Cross-Entropy and Dice loss.
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

        return bce + dice_loss