import json
import random
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent

TRAIN_DIR = ROOT / "data" / "train_images"
VAL_DIR = ROOT / "data" / "validation_images"
TEST_DIR = ROOT / "data" / "test_images"

TRAIN_DIR.mkdir(parents=True, exist_ok=True)
VAL_DIR.mkdir(parents=True, exist_ok=True)
TEST_DIR.mkdir(parents=True, exist_ok=True)


API_URL = (
    "https://api.github.com/repos/"
    "jp17245/DAVROS/contents/INRIAPerson/Train/pos"
)


print("Getting INRIA image list...")

request = urllib.request.Request(
    API_URL,
    headers={
        "User-Agent": "INRIA-subset-downloader"
    }
)

with urllib.request.urlopen(request) as response:
    files = json.load(response)


# Keep only image files
images = [
    file
    for file in files
    if file["name"].lower().endswith(
        (".png", ".jpg", ".jpeg")
    )
]

print(f"Images available: {len(images)}")


# Reproducible random split
random.seed(42)
random.shuffle(images)


TRAIN_COUNT = 140
VAL_COUNT = 20
TEST_COUNT = 20

TOTAL_NEEDED = (
    TRAIN_COUNT
    + VAL_COUNT
    + TEST_COUNT
)

if len(images) < TOTAL_NEEDED:
    raise RuntimeError(
        f"Only {len(images)} images available, "
        f"but {TOTAL_NEEDED} are required."
    )


train_images = images[:TRAIN_COUNT]

val_images = images[
    TRAIN_COUNT:
    TRAIN_COUNT + VAL_COUNT
]

test_images = images[
    TRAIN_COUNT + VAL_COUNT:
    TOTAL_NEEDED
]


def download(images, folder, split_name):

    print()
    print(
        f"Downloading {len(images)} "
        f"{split_name} images..."
    )

    for number, image in enumerate(images, start=1):

        destination = folder / image["name"]

        print(
            f"[{number}/{len(images)}] "
            f"{image['name']}"
        )

        urllib.request.urlretrieve(
            image["download_url"],
            destination
        )


download(
    train_images,
    TRAIN_DIR,
    "training"
)

download(
    val_images,
    VAL_DIR,
    "validation"
)

download(
    test_images,
    TEST_DIR,
    "test"
)


print()
print("==============================")
print("DOWNLOAD COMPLETE")
print("==============================")
print(f"Training:   {len(train_images)}")
print(f"Validation: {len(val_images)}")
print(f"Testing:    {len(test_images)}")
