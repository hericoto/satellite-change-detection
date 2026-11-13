
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

from model import SiameseUNet


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def load_image(path):
    img = np.array(
        Image.open(path).convert("RGB")
    ).astype(np.float32) / 255.0

    tensor = torch.from_numpy(
        img
    ).permute(2, 0, 1).unsqueeze(0)

    return tensor.to(DEVICE), img


def main():
    a_path = Path("data/processed/test/A")
    b_path = Path("data/processed/test/B")
    m_path = Path("data/processed/test/label")

    files = sorted(a_path.glob("*.png"))

    model = SiameseUNet().to(DEVICE)

    model.load_state_dict(
        torch.load(
            "weights/best_model.pth",
            map_location=DEVICE
        )
    )

    model.eval()

    output_dir = Path("results/predictions")
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for path in files[:10]:
            name = path.name

            a, img_a = load_image(a_path / name)
            b, img_b = load_image(b_path / name)

            mask = np.array(
                Image.open(m_path / name).convert("L")
            ) > 127

            prediction = (
                torch.sigmoid(model(a, b)) > 0.5
            ).squeeze().cpu().numpy()

            fig, axes = plt.subplots(1, 4, figsize=(16, 4))

            axes[0].imshow(img_a)
            axes[0].set_title("Image A")

            axes[1].imshow(img_b)
            axes[1].set_title("Image B")

            axes[2].imshow(mask, cmap="gray")
            axes[2].set_title("Ground Truth")

            axes[3].imshow(prediction, cmap="gray")
            axes[3].set_title("Prediction")

            for ax in axes:
                ax.axis("off")

            plt.tight_layout()
            plt.savefig(output_dir / name)
            plt.close()

    print("Predictions saved!")


if __name__ == "__main__":
    main()
