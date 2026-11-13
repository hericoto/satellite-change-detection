
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class ChangeDetectionDataset(Dataset):
    def __init__(self, root, split):
        self.root = Path(root)
        self.split = split

        self.a_dir = self.root / split / "A"
        self.b_dir = self.root / split / "B"
        self.m_dir = self.root / split / "label"

        self.files = sorted(
            p.name for p in self.a_dir.glob("*.png")
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]

        a = np.array(
            Image.open(self.a_dir / name).convert("RGB")
        ).astype(np.float32) / 255.0

        b = np.array(
            Image.open(self.b_dir / name).convert("RGB")
        ).astype(np.float32) / 255.0

        mask = np.array(
            Image.open(self.m_dir / name).convert("L")
        )

        a = torch.from_numpy(a).permute(2, 0, 1)
        b = torch.from_numpy(b).permute(2, 0, 1)

        mask = torch.from_numpy(
            (mask > 127).astype(np.float32)
        ).unsqueeze(0)

        return a, b, mask
