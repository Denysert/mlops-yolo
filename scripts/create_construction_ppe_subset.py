from pathlib import Path
import random
import shutil
import yaml

SOURCE = Path("datasets/construction-ppe")
TARGET = Path("datasets/construction-ppe-small")

random.seed(42)

if TARGET.exists():
    raise SystemExit("Target dataset already exists.")

def get_images(split):
    return sorted(
        p for p in (SOURCE / "images" / split).iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

train = random.sample(get_images("train"), 100)
validation = random.sample(get_images("val"), 60)

splits = {
    "train": ("train", train),
    "val": ("val", validation[:30]),
    "test": ("val", validation[30:])
}

for split, (original, images) in splits.items():
    image_dir = TARGET / "images" / split
    label_dir = TARGET / "labels" / split

    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    for image in images:
        label = SOURCE / "labels" / original / (
            image.stem + ".txt"
        )

        shutil.copy2(image, image_dir / image.name)
        shutil.copy2(label, label_dir / label.name)

    print(f"{split}: {len(images)} images copied")

with open(SOURCE / "data.yaml") as f:
    config = yaml.safe_load(f)

config["path"] = str(TARGET.resolve())
config["train"] = "images/train"
config["val"] = "images/val"
config["test"] = "images/test"

with open(TARGET / "data.yaml", "w") as f:
    yaml.safe_dump(config, f, sort_keys=False)

shutil.copy2(SOURCE / "LICENSE.txt", TARGET / "LICENSE.txt")

(TARGET / "README.md").write_text(
    "Adapted from Ultralytics Construction-PPE Dataset, "
    "Ultralytics team, November 2026, CC BY 4.0.\n"
    "https://docs.ultralytics.com/datasets/detect/construction-ppe\n"
    "https://creativecommons.org/licenses/by/4.0/\n"
    "Changes: sampled 100 train, 30 val and 30 test images.\n",
    encoding="utf-8"
)

print("Dataset created successfully!")
