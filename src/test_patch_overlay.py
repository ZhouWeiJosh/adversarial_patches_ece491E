from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as TF

from patch_utils import (
    create_random_patch,
    resize_patch,
    place_patch_on_image
)


ROOT = Path(__file__).resolve().parent.parent

IMAGE_PATH = (
    ROOT
    / "data"
    / "test_images"
    / "crop001001.png"
)

OUTPUT_PATH = (
    ROOT
    / "results"
    / "images"
    / "patch_overlay.png"
)


def main():

    print("Loading image...")

    image = Image.open(
        IMAGE_PATH
    ).convert("RGB")

    image_tensor = TF.to_tensor(image)

    _, height, width = image_tensor.shape

    print(f"Image size: {width} x {height}")

    # -----------------------------------------
    # Create random patch
    # -----------------------------------------

    patch = create_random_patch(
        patch_size=100
    )

    # Make patch roughly 25% of image width
    target_size = int(width * 0.25)

    patch = resize_patch(
        patch,
        target_size
    )

    # -----------------------------------------
    # Put patch in center of image
    # -----------------------------------------

    center_x = width // 2
    center_y = height // 2

    patched_image = place_patch_on_image(
        image_tensor,
        patch,
        center_x,
        center_y
    )

    # -----------------------------------------
    # Save
    # -----------------------------------------

    output = TF.to_pil_image(
        patched_image
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output.save(OUTPUT_PATH)

    print(f"Saved:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
