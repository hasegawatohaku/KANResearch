import torch
import numpy as np

from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset

class BUSISegmentationDataset(Dataset):
    """
    BUSI binary lesion-segmentation dataset.

    Expected directory structure:

        Dataset_BUSI_with_GT/
            benign/
                benign (1).png
                benign (1)_mask.png
                ...
            malignant/
                malignant (1).png
                malignant (1)_mask.png
                ...
            normal/
                ...

    Normal images are not used because they do not contain a lesion mask.
    If several masks exist for one image, they are merged into one binary mask.
    """

    def __init__(self, root, image_size=256):
        self.root = Path(root)
        self.image_size = image_size
        self.samples = []

        for class_name in ["benign", "malignant"]:
            class_dir = self.root / class_name

            if not class_dir.exists():
                continue

            for image_path in sorted(class_dir.glob("*.png")):
                # Mask files are not input images.
                if "_mask" in image_path.stem:
                    continue

                masks = sorted(
                    class_dir.glob(f"{image_path.stem}_mask*.png")
                )

                if len(masks) == 0:
                    continue

                self.samples.append(
                    (image_path, masks)
                )

        if len(self.samples) == 0:
            raise RuntimeError(
                "BUSI dataset was not found or no image/mask pairs were found.\n"
                f"Expected root: {self.root}\n"
                "Download BUSI and extract Dataset_BUSI_with_GT/ here."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_paths = self.samples[index]

        image = Image.open(image_path).convert("L")
        image = image.resize(
            (self.image_size, self.image_size),
            Image.Resampling.BILINEAR,
        )

        image = torch.from_numpy(
            np.asarray(image, dtype=np.float32)
        ) / 255.0
        image = image.unsqueeze(0)

        mask = np.zeros(
            (self.image_size, self.image_size),
            dtype=np.uint8,
        )

        for mask_path in mask_paths:
            current_mask = Image.open(mask_path).convert("L")
            current_mask = current_mask.resize(
                (self.image_size, self.image_size),
                Image.Resampling.NEAREST,
            )
            current_mask = np.asarray(current_mask)
            mask = np.maximum(mask, current_mask)

        mask = (mask > 0).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask