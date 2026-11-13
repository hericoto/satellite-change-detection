
from pathlib import Path
from PIL import Image
import csv

RAW = Path("data/raw/LEVIR-CD")
OUT = Path("data/processed")

PATCH_SIZE = 256
PATCHES_PER_IMAGE = 8

SPLITS = ["train", "val", "test"]


def get_images(folder):
    extensions = {".png", ".jpg", ".jpeg"}
    return sorted(
        p for p in folder.iterdir()
        if p.suffix.lower() in extensions
    )


def main():
    manifest = []

    for split in SPLITS:
        for subfolder in ["A", "B", "label"]:
            (OUT / split / subfolder).mkdir(
                parents=True, exist_ok=True
            )

        files_a = get_images(RAW / split / "A")
        files_b = get_images(RAW / split / "B")
        files_m = get_images(RAW / split / "label")

        if not (len(files_a) == len(files_b) == len(files_m)):
            raise ValueError(f"Different file counts in {split}")

        for a, b, mask in zip(files_a, files_b, files_m):
            if a.stem != b.stem or a.stem != mask.stem:
                raise ValueError(f"File mismatch: {a.stem}")

            img_a = Image.open(a).convert("RGB")
            img_b = Image.open(b).convert("RGB")
            img_m = Image.open(mask).convert("L")

            if not (
                img_a.size == img_b.size == img_m.size
            ):
                raise ValueError(f"Different dimensions: {a.stem}")

            width, height = img_a.size

            if width != 1024 or height != 1024:
                raise ValueError(f"Unexpected image size: {a}")

            # Выбираем 8 равномерно распределённых
            # координат из сетки 4 x 4.
            positions = [
                (0, 0), (0, 256),
                (256, 0), (256, 256),
                (0, 512), (256, 512),
                (512, 0), (512, 256)
            ]

            for idx, (x, y) in enumerate(positions):
                if idx >= PATCHES_PER_IMAGE:
                    break

                name = f"{a.stem}_{idx:02d}.png"
                coords = (x, y, x + PATCH_SIZE, y + PATCH_SIZE)

                img_a.crop(coords).save(
                    OUT / split / "A" / name
                )
                img_b.crop(coords).save(
                    OUT / split / "B" / name
                )
                img_m.crop(coords).save(
                    OUT / split / "label" / name
                )

                manifest.append({
                    "split": split,
                    "filename": name,
                    "source": a.stem,
                    "x": x,
                    "y": y
                })

    with open(
        OUT / "manifest.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split", "filename", "source", "x", "y"
            ]
        )
        writer.writeheader()
        writer.writerows(manifest)

    print("Preparation completed!")
    print("Total pairs:", len(manifest))


if __name__ == "__main__":
    main()
