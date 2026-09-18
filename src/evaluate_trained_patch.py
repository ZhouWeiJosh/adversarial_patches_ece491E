from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as TF

from darknet import Darknet
from utils import do_detect

from patch_utils import create_random_patch


# ==================================================
# Paths
# ==================================================

ROOT = Path(__file__).resolve().parent.parent

CFG_PATH = ROOT / "cfg" / "yolo.cfg"
WEIGHTS_PATH = ROOT / "weights" / "yolo.weights"

IMAGE_PATH = (
    ROOT
    / "data"
    / "test_images"
    / "crop001001.png"
)

TRAINED_PATCH_PATH = (
    ROOT
    / "patches"
    / "one_image_patch.png"
)

RESULTS_DIR = (
    ROOT
    / "results"
    / "images"
)


# ==================================================
# Settings
# ==================================================

YOLO_SIZE = 416

CONF_THRESHOLD = 0.4
NMS_THRESHOLD = 0.4

# IMPORTANT:
# This should match train_one_patch.py
PATCH_SIZE = 125

# IMPORTANT:
# This should match train_one_patch.py
TORSO_POSITION = 0.40

USE_CUDA = torch.cuda.is_available()


# ==================================================
# Get person confidence
# ==================================================

def get_best_person_confidence(boxes):

    best_confidence = 0.0

    for box in boxes:

        class_id = int(box[6])

        # COCO person = class 0
        if class_id != 0:
            continue

        objectness = box[4]
        class_confidence = box[5]

        if torch.is_tensor(objectness):
            objectness = (
                objectness
                .detach()
                .cpu()
                .item()
            )
        else:
            objectness = float(objectness)

        if torch.is_tensor(class_confidence):
            class_confidence = (
                class_confidence
                .detach()
                .cpu()
                .item()
            )
        else:
            class_confidence = float(
                class_confidence
            )

        confidence = (
            objectness
            *
            class_confidence
        )

        best_confidence = max(
            best_confidence,
            confidence
        )

    return best_confidence


# ==================================================
# Detection
# ==================================================

def run_detection(
    model,
    image
):

    boxes = do_detect(
        model,
        image,
        CONF_THRESHOLD,
        NMS_THRESHOLD,
        USE_CUDA
    )

    confidence = (
        get_best_person_confidence(
            boxes
        )
    )

    return boxes, confidence


# ==================================================
# Find best person
# ==================================================

def get_best_person_box(boxes):

    person_boxes = []

    for box in boxes:

        class_id = int(box[6])

        if class_id == 0:
            person_boxes.append(box)

    if len(person_boxes) == 0:

        return None

    def confidence(box):

        objectness = box[4]
        class_confidence = box[5]

        if torch.is_tensor(objectness):
            objectness = (
                objectness
                .detach()
                .cpu()
                .item()
            )

        if torch.is_tensor(class_confidence):
            class_confidence = (
                class_confidence
                .detach()
                .cpu()
                .item()
            )

        return (
            float(objectness)
            *
            float(class_confidence)
        )

    return max(
        person_boxes,
        key=confidence
    )


# ==================================================
# Place patch relative to person
# ==================================================

def apply_patch_on_person(
    image_tensor,
    patch,
    person_box
):

    """
    image_tensor:
        [3, H, W]

    patch:
        [3, P, P]

    person_box:
        normalized YOLO bbox:
        [center_x, center_y, width, height, ...]
    """

    patched = image_tensor.clone()

    _, image_h, image_w = (
        patched.shape
    )

    # Convert YOLO normalized bbox
    # into pixel coordinates
    person_center_x = (
        float(person_box[0])
        *
        image_w
    )

    person_center_y = (
        float(person_box[1])
        *
        image_h
    )

    person_height = (
        float(person_box[3])
        *
        image_h
    )

    # Top of person's bbox
    person_top = (
        person_center_y
        -
        person_height / 2
    )

    # Patch position
    patch_center_x = (
        person_center_x
    )

    patch_center_y = (
        person_top
        +
        TORSO_POSITION
        *
        person_height
    )

    patch_h = patch.shape[1]
    patch_w = patch.shape[2]

    x1 = int(
        patch_center_x
        -
        patch_w / 2
    )

    y1 = int(
        patch_center_y
        -
        patch_h / 2
    )

    x2 = x1 + patch_w
    y2 = y1 + patch_h

    # Keep inside image
    x1 = max(
        0,
        x1
    )

    y1 = max(
        0,
        y1
    )

    x2 = min(
        image_w,
        x2
    )

    y2 = min(
        image_h,
        y2
    )

    patch_width = (
        x2 - x1
    )

    patch_height = (
        y2 - y1
    )

    patched[
        :,
        y1:y2,
        x1:x2
    ] = patch[
        :,
        0:patch_height,
        0:patch_width
    ]

    return patched


# ==================================================
# Save tensor image
# ==================================================

def save_tensor_image(
    tensor,
    path
):

    image = TF.to_pil_image(
        tensor
        .detach()
        .cpu()
        .clamp(0, 1)
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    image.save(path)


# ==================================================
# Main
# ==================================================

def main():

    print("======================================")
    print("EVALUATE TRAINED PATCH")
    print("======================================")

    print(
        f"Device: "
        f"{'CUDA' if USE_CUDA else 'CPU'}"
    )

    print()

    # ------------------------------------------------
    # Check patch
    # ------------------------------------------------

    if not TRAINED_PATCH_PATH.exists():

        print(
            "ERROR: trained patch not found:"
        )

        print(
            TRAINED_PATCH_PATH
        )

        return

    # ------------------------------------------------
    # Load YOLO
    # ------------------------------------------------

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
    print()

    # ------------------------------------------------
    # Load image
    # ------------------------------------------------

    original_image = Image.open(
        IMAGE_PATH
    ).convert("RGB")

    print(
        f"Original image size: "
        f"{original_image.size}"
    )

    image = original_image.resize(
        (
            YOLO_SIZE,
            YOLO_SIZE
        )
    )

    print(
        f"YOLO input size: "
        f"{image.size}"
    )

    image_tensor = TF.to_tensor(
        image
    )

    # =================================================
    # 1. CLEAN
    # =================================================

    print()
    print("Running CLEAN detection...")

    clean_boxes, clean_confidence = (
        run_detection(
            model,
            image
        )
    )

    # ------------------------------------------------
    # Get person bbox from CLEAN image
    # ------------------------------------------------

    person_box = (
        get_best_person_box(
            clean_boxes
        )
    )

    if person_box is None:

        print(
            "ERROR: No person detected."
        )

        return

    print()
    print("Person found.")

    print(
        "Bounding box:",
        person_box[:4]
    )

    # =================================================
    # 2. RANDOM PATCH
    # =================================================

    print()
    print(
        "Running RANDOM PATCH detection..."
    )

    random_patch = create_random_patch(
        patch_size=PATCH_SIZE
    )

    random_patched_tensor = (
        apply_patch_on_person(
            image_tensor,
            random_patch,
            person_box
        )
    )

    random_patched_image = (
        TF.to_pil_image(
            random_patched_tensor
        )
    )

    (
        random_boxes,
        random_confidence
    ) = run_detection(
        model,
        random_patched_image
    )

    # =================================================
    # 3. TRAINED PATCH
    # =================================================

    print(
        "Running TRAINED PATCH detection..."
    )

    trained_patch_image = (
        Image.open(
            TRAINED_PATCH_PATH
        )
        .convert("RGB")
    )

    trained_patch = TF.to_tensor(
        trained_patch_image
    )

    # ------------------------------------------------
    # Make absolutely sure it matches training size
    # ------------------------------------------------

    if (
        trained_patch.shape[1]
        != PATCH_SIZE
        or
        trained_patch.shape[2]
        != PATCH_SIZE
    ):

        trained_patch_image = (
            trained_patch_image.resize(
                (
                    PATCH_SIZE,
                    PATCH_SIZE
                )
            )
        )

        trained_patch = (
            TF.to_tensor(
                trained_patch_image
            )
        )

    trained_patched_tensor = (
        apply_patch_on_person(
            image_tensor,
            trained_patch,
            person_box
        )
    )

    trained_patched_image = (
        TF.to_pil_image(
            trained_patched_tensor
        )
    )

    (
        trained_boxes,
        trained_confidence
    ) = run_detection(
        model,
        trained_patched_image
    )

    # =================================================
    # Save images
    # =================================================

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    image.save(
        RESULTS_DIR
        / "eval_clean.png"
    )

    save_tensor_image(
        random_patched_tensor,
        RESULTS_DIR
        / "eval_random_patch.png"
    )

    save_tensor_image(
        trained_patched_tensor,
        RESULTS_DIR
        / "eval_trained_patch.png"
    )

    # =================================================
    # Results
    # =================================================

    print()
    print("======================================")
    print("RESULTS")
    print("======================================")

    print(
        f"Clean person confidence:  "
        f"{clean_confidence:.4f}"
    )

    print(
        f"Random patch confidence:  "
        f"{random_confidence:.4f}"
    )

    print(
        f"Trained patch confidence: "
        f"{trained_confidence:.4f}"
    )

    print()

    clean_drop_random = (
        clean_confidence
        -
        random_confidence
    )

    clean_drop_trained = (
        clean_confidence
        -
        trained_confidence
    )

    print(
        f"Random patch confidence drop: "
        f"{clean_drop_random:.4f}"
    )

    print(
        f"Trained patch confidence drop: "
        f"{clean_drop_trained:.4f}"
    )

    print()

    # =================================================
    # Interpretation
    # =================================================

    if (
        trained_confidence
        <
        random_confidence
    ):

        print("SUCCESS:")

        print(
            "The trained patch reduced "
            "person confidence more than "
            "the random patch."
        )

    else:

        print("NOTE:")

        print(
            "The trained patch did not "
            "outperform the random patch."
        )

    print()

    if (
        trained_confidence
        <
        CONF_THRESHOLD
    ):

        print(
            f"At threshold "
            f"{CONF_THRESHOLD:.2f}, "
            f"the trained patch suppresses "
            f"the person detection."
        )

    else:

        print(
            f"At threshold "
            f"{CONF_THRESHOLD:.2f}, "
            f"YOLO still detects the person."
        )

    print()

    print(
        "Saved evaluation images to:"
    )

    print(
        RESULTS_DIR
    )


if __name__ == "__main__":
    main()