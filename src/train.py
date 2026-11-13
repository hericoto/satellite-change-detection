
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import SiameseUNet
from dataset import ChangeDetectionDataset


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

EPOCHS = 20
BATCH_SIZE = 8
LR = 0.001

Path("weights").mkdir(exist_ok=True)
Path("outputs").mkdir(exist_ok=True)


def dice_loss(logits, targets):
    probs = torch.sigmoid(logits)

    intersection = (probs * targets).sum(
        dim=(1, 2, 3)
    )

    denominator = (
        probs.sum(dim=(1, 2, 3))
        + targets.sum(dim=(1, 2, 3))
    )

    dice = (2 * intersection + 1e-6) / (
        denominator + 1e-6
    )

    return 1 - dice.mean()


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for a, b, mask in loader:
            a, b, mask = (
                a.to(DEVICE),
                b.to(DEVICE),
                mask.to(DEVICE)
            )

            logits = model(a, b)

            loss = criterion(logits, mask)
            loss += dice_loss(logits, mask)

            total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def main():
    train_ds = ChangeDetectionDataset(
        "data/processed", "train"
    )

    val_ds = ChangeDetectionDataset(
        "data/processed", "val"
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model = SiameseUNet().to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR
    )

    best_loss = float("inf")
    train_history = []
    val_history = []

    print("Device:", DEVICE)
    print("Train samples:", len(train_ds))
    print("Validation samples:", len(val_ds))

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0

        loop = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{EPOCHS}"
        )

        for a, b, mask in loop:
            a = a.to(DEVICE)
            b = b.to(DEVICE)
            mask = mask.to(DEVICE)

            optimizer.zero_grad()

            logits = model(a, b)

            loss = criterion(logits, mask)
            loss += dice_loss(logits, mask)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            loop.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        train_loss = running_loss / len(train_loader)

        val_loss = evaluate(
            model, val_loader, criterion
        )

        train_history.append(train_loss)
        val_history.append(val_loss)

        print(
            f"Epoch {epoch + 1}: "
            f"train={train_loss:.4f}, "
            f"val={val_loss:.4f}"
        )

        if val_loss < best_loss:
            best_loss = val_loss

            torch.save(
                model.state_dict(),
                "weights/best_model.pth"
            )

    plt.figure(figsize=(9, 5))
    plt.plot(train_history, label="Train")
    plt.plot(val_history, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training history")
    plt.legend()
    plt.grid()
    plt.savefig("outputs/training_history.png")
    plt.close()

    print("Training completed!")


if __name__ == "__main__":
    main()
