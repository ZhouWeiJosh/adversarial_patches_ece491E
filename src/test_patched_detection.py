from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as TF

from darknet import Darknet
from utils import do_detect

from patch_utils import (
    create_random_patch,
    resize_patch,
    place_patch_on_image
)


ROOT = Path(__file__).resolve().parent.parent

CFG_PATH = ROOT / "cfg" / "yolo.cfg"

WEIGHTS_PATH = (
    ROOT
    / "weights"
    / "yolo.weights"
)

IMAGE_PATH = (
    ROOT
    / "data"
    / "test_images"
    / "crop001001.png"
)


CONF_THRESHOLD = 0.4
NMS_THRESHOLD = 0.4

USE_CUDA = torch.cuda.is_available()


def get_person_confidence(boxes):

    best_confidence = 0.0

    for box in boxes:

        class_id = int(box[6])

        # COCO person = class 0
        if class_id != 0:
            continue

        objectness = float(box[4])

        class_confidence = float(box[5])

        confidence = (
            objectness
            * class_confidence
        )

        best_confidence = max(
            best_confidence,
            confidence
        )

    return best_confidence


def main():

    # -----------------------------------------
    # YOLO
    # -----------------------------------------

    print("Loading YOLOv2...")

    model = Darknet(
        str(CFG_PATH)
    )

    model.load_weights(
        str(WEIGHTS_PATH)
    )

    model.eval()

    if USE_CUDA:
        model.cuda()

    print("YOLO loaded.")

    # -----------------------------------------
    # Image
    # -----------------------------------------

    original_image = Image.open(
        IMAGE_PATH
    ).convert("RGB")

    print(
        f"Original image size: "
        f"{original_image.size}"
    )

    # Resize to YOLOv2 input size
    image = original_image.resize(
        (
            model.width,
            model.height
        )
    )

    print(
        f"YOLO input size: "
        f"{image.size}"
    )

    # -----------------------------------------
    # Clean detection
    # -----------------------------------------

    clean_boxes = do_detect(
        model,
        image,
        CONF_THRESHOLD,
        NMS_THRESHOLD,
        USE_CUDA
    )

    clean_conf = get_person_confidence(
        clean_boxes
    )

    # -----------------------------------------
    # Random patch
    # -----------------------------------------

    tensor = TF.to_tensor(image)

    _, h, w = tensor.shape

    patch = create_random_patch(100)

    patch = resize_patch(
        patch,
        int(w * 0.25)
    )

    patched_tensor = place_patch_on_image(
        tensor,
        patch,
        w // 2,
        h // 2
    )

    patched_image = TF.to_pil_image(
        patched_tensor
    )

    print(
        f"Patched image size: "
        f"{patched_image.size}"
    )

    # -----------------------------------------
    # Patched detection
    # -----------------------------------------

    patch_boxes = do_detect(
        model,
        patched_image,
        CONF_THRESHOLD,
        NMS_THRESHOLD,
        USE_CUDA
    )

    patch_conf = get_person_confidence(
        patch_boxes
    )

    # -----------------------------------------
    # Results
    # -----------------------------------------

    print()
    print("==============================")
    print("RESULTS")
    print("==============================")

    print(
        f"Clean person confidence: "
        f"{clean_conf:.4f}"
    )

    print(
        f"Random patch confidence: "
        f"{patch_conf:.4f}"
    )


if __name__ == "__main__":
    main()