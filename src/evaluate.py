
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader

from model import SiameseUNet
from dataset import ChangeDetectionDataset


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def main():
    dataset = ChangeDetectionDataset(
        "data/processed", "test"
    )

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False
    )

    model = SiameseUNet().to(DEVICE)

    model.load_state_dict(
        torch.load(
            "weights/best_model.pth",
            map_location=DEVICE
        )
    )

    model.eval()

    tp = fp = fn = tn = 0

    with torch.no_grad():
        for a, b, mask in loader:
            a = a.to(DEVICE)
            b = b.to(DEVICE)

            prediction = torch.sigmoid(
                model(a, b)
            ) > 0.5

            target = mask.to(DEVICE) > 0.5

            tp += (prediction & target).sum().item()
            fp += (prediction & ~target).sum().item()
            fn += (~prediction & target).sum().item()
            tn += (~prediction & ~target).sum().item()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    f1 = (
        2 * precision * recall /
        max(precision + recall, 1e-10)
    )

    iou = tp / max(tp + fp + fn, 1)

    accuracy = (
        (tp + tn) / max(tp + tn + fp + fn, 1)
    )

    results = {
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "IoU": iou,
        "Accuracy": accuracy
    }

    print("\nTEST RESULTS")

    for key, value in results.items():
        print(f"{key}: {value:.4f}")

    Path("outputs").mkdir(exist_ok=True)

    pd.DataFrame(
        [results]
    ).to_csv(
        "outputs/metrics.csv",
        index=False
    )


if __name__ == "__main__":
    main()
