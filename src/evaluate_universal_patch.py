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

WEIGHTS_PATH = (
    ROOT
    / "weights"
    / "yolo.weights"
)

DATA_DIR = (
    ROOT
    / "data"
    / "validation_images"
)

UNIVERSAL_PATCH_PATH = (
    ROOT
    / "patches"
    / "universal_patch_cls.png"
)


# ==================================================
# Settings
# ==================================================

YOLO_SIZE = 416

PATCH_SIZE = 125

TORSO_POSITION = 0.40

CONF_THRESHOLD = 0.4

NMS_THRESHOLD = 0.4


# Evaluate up to 50 images from validation_image.
NUM_VALIDATION_IMAGES = 20


USE_CUDA = torch.cuda.is_available()


# ==================================================
# Helper functions
# ==================================================

def get_value(value):

    if torch.is_tensor(value):

        return (
            value
            .detach()
            .cpu()
            .item()
        )

    return float(value)


def get_best_person_confidence(boxes):

    best_confidence = 0.0

    for box in boxes:

        class_id = int(box[6])

        if class_id != 0:
            continue

        objectness = get_value(
            box[4]
        )

        class_confidence = get_value(
            box[5]
        )

        final_confidence = (
            objectness
            *
            class_confidence
        )

        best_confidence = max(
            best_confidence,
            final_confidence
        )

    return best_confidence


def get_best_person_box(boxes):

    person_boxes = []

    for box in boxes:

        class_id = int(box[6])

        if class_id == 0:
            person_boxes.append(box)

    if len(person_boxes) == 0:
        return None

    def confidence(box):

        objectness = get_value(
            box[4]
        )

        class_confidence = get_value(
            box[5]
        )

        return (
            objectness
            *
            class_confidence
        )

    return max(
        person_boxes,
        key=confidence
    )


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
    """

    patched = image_tensor.clone()

    _, image_h, image_w = (
        patched.shape
    )

    # ----------------------------------------------
    # Person bbox
    # ----------------------------------------------

    person_center_x = (
        get_value(
            person_box[0]
        )
        *
        image_w
    )

    person_center_y = (
        get_value(
            person_box[1]
        )
        *
        image_h
    )

    person_height = (
        get_value(
            person_box[3]
        )
        *
        image_h
    )

    person_top = (
        person_center_y
        -
        person_height / 2
    )

    # ----------------------------------------------
    # Patch position
    # ----------------------------------------------

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

    # ----------------------------------------------
    # Clamp to image boundaries
    # ----------------------------------------------

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

    # ----------------------------------------------
    # Apply patch
    # ----------------------------------------------

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


def find_images(directory):

    extensions = [
        "*.png",
        "*.jpg",
        "*.jpeg",
        "*.PNG",
        "*.JPG",
        "*.JPEG",
    ]

    images = []

    for extension in extensions:

        images.extend(
            directory.glob(extension)
        )

    return sorted(images)


# ==================================================
# Main
# ==================================================

def main():

    print(
        "======================================"
    )

    print(
        "EVALUATE UNIVERSAL PATCH"
    )

    print(
        "======================================"
    )

    print(
        f"Device: "
        f"{'CUDA' if USE_CUDA else 'CPU'}"
    )

    print()

    # ==================================================
    # Check files
    # ==================================================

    if not DATA_DIR.exists():

        print(
            "ERROR: dataset directory not found:"
        )

        print(
            DATA_DIR
        )

        return

    if not UNIVERSAL_PATCH_PATH.exists():

        print(
            "ERROR: universal patch not found:"
        )

        print(
            UNIVERSAL_PATCH_PATH
        )

        return

    # ==================================================
    # Find images
    # ==================================================

    all_images = (
        find_images(
            DATA_DIR
        )
    )

    print(
        f"Total images found: "
        f"{len(all_images)}"
    )

    validation_images = (
        all_images[
            :NUM_VALIDATION_IMAGES
        ]
    )

    print(
        f"Validation images selected: "
        f"{len(validation_images)}"
    )

    print()

    # ==================================================
    # Load YOLO
    # ==================================================

    print(
        "Loading YOLOv2..."
    )

    model = Darknet(
        str(CFG_PATH)
    )

    model.load_weights(
        str(WEIGHTS_PATH)
    )

    model.eval()

    if USE_CUDA:
        model.cuda()

    print(
        "YOLO loaded."
    )

    print()

    # ==================================================
    # Load universal patch
    # ==================================================

    universal_patch_image = (
        Image.open(
            UNIVERSAL_PATCH_PATH
        )
        .convert("RGB")
    )

    universal_patch_image = (
        universal_patch_image.resize(
            (
                PATCH_SIZE,
                PATCH_SIZE
            )
        )
    )

    universal_patch = TF.to_tensor(
        universal_patch_image
    )

    # ==================================================
    # Statistics
    # ==================================================

    total_clean_conf = 0.0
    total_random_conf = 0.0
    total_universal_conf = 0.0

    clean_detected = 0
    random_detected = 0
    universal_detected = 0

    evaluated_images = 0
    skipped_images = 0


    # ==================================================
    # Evaluation loop
    # ==================================================

    print(
        "======================================"
    )

    print(
        "STARTING VALIDATION"
    )

    print(
        "======================================"
    )

    print()

    for index, image_path in enumerate(
        validation_images,
        start=1
    ):

        print(
            f"--------------------------------------"
        )

        print(
            f"Image "
            f"{index}/"
            f"{len(validation_images)}: "
            f"{image_path.name}"
        )

        print(
            f"--------------------------------------"
        )

        # ------------------------------------------
        # Load image
        # ------------------------------------------

        try:

            image = (
                Image.open(
                    image_path
                )
                .convert("RGB")
            )

        except Exception as error:

            print(
                f"Could not load image: "
                f"{error}"
            )

            skipped_images += 1

            continue

        image = image.resize(
            (
                YOLO_SIZE,
                YOLO_SIZE
            )
        )

        image_tensor = TF.to_tensor(
            image
        )

        # ==========================================
        # CLEAN
        # ==========================================

        clean_boxes, clean_conf = (
            run_detection(
                model,
                image
            )
        )

        person_box = (
            get_best_person_box(
                clean_boxes
            )
        )

        # We need a clean person bbox
        # so we know where to put patches.
        if person_box is None:

            print(
                "No clean person detection."
            )

            print(
                "Skipping image."
            )

            skipped_images += 1

            continue

        # ==========================================
        # RANDOM PATCH
        # ==========================================

        random_patch = (
            create_random_patch(
                patch_size=PATCH_SIZE
            )
        )

        random_patched_tensor = (
            apply_patch_on_person(
                image_tensor,
                random_patch,
                person_box
            )
        )

        random_image = (
            TF.to_pil_image(
                random_patched_tensor
            )
        )

        (
            random_boxes,
            random_conf
        ) = run_detection(
            model,
            random_image
        )

        # ==========================================
        # UNIVERSAL PATCH
        # ==========================================

        universal_patched_tensor = (
            apply_patch_on_person(
                image_tensor,
                universal_patch,
                person_box
            )
        )

        universal_image = (
            TF.to_pil_image(
                universal_patched_tensor
            )
        )

        (
            universal_boxes,
            universal_conf
        ) = run_detection(
            model,
            universal_image
        )

        # ==========================================
        # Save statistics
        # ==========================================

        total_clean_conf += (
            clean_conf
        )

        total_random_conf += (
            random_conf
        )

        total_universal_conf += (
            universal_conf
        )

        if (
            clean_conf
            >=
            CONF_THRESHOLD
        ):

            clean_detected += 1

        if (
            random_conf
            >=
            CONF_THRESHOLD
        ):

            random_detected += 1

        if (
            universal_conf
            >=
            CONF_THRESHOLD
        ):

            universal_detected += 1

        evaluated_images += 1

        # ==========================================
        # Print image result
        # ==========================================

        print(
            f"CLEAN:     "
            f"{clean_conf:.4f}"
        )

        print(
            f"RANDOM:    "
            f"{random_conf:.4f}"
        )

        print(
            f"UNIVERSAL: "
            f"{universal_conf:.4f}"
        )

        print()

    # ==================================================
    # Final statistics
    # ==================================================

    print()
    print(
        "======================================"
    )

    print(
        "FINAL RESULTS"
    )

    print(
        "======================================"
    )

    print()

    print(
        f"Images evaluated: "
        f"{evaluated_images}"
    )

    print(
        f"Images skipped:   "
        f"{skipped_images}"
    )

    print()

    if evaluated_images == 0:

        print(
            "No images were successfully "
            "evaluated."
        )

        return

    # ----------------------------------------------
    # Average confidence
    # ----------------------------------------------

    average_clean = (
        total_clean_conf
        /
        evaluated_images
    )

    average_random = (
        total_random_conf
        /
        evaluated_images
    )

    average_universal = (
        total_universal_conf
        /
        evaluated_images
    )

    print(
        "AVERAGE PERSON CONFIDENCE"
    )

    print(
        f"Clean:     "
        f"{average_clean:.4f}"
    )

    print(
        f"Random:    "
        f"{average_random:.4f}"
    )

    print(
        f"Universal: "
        f"{average_universal:.4f}"
    )

    print()

    # ----------------------------------------------
    # Detection rate
    # ----------------------------------------------

    clean_detection_rate = (
        clean_detected
        /
        evaluated_images
        *
        100
    )

    random_detection_rate = (
        random_detected
        /
        evaluated_images
        *
        100
    )

    universal_detection_rate = (
        universal_detected
        /
        evaluated_images
        *
        100
    )

    print(
        f"DETECTION RATE "
        f"(threshold = "
        f"{CONF_THRESHOLD:.2f})"
    )

    print(
        f"Clean:     "
        f"{clean_detection_rate:.2f}%"
    )

    print(
        f"Random:    "
        f"{random_detection_rate:.2f}%"
    )

    print(
        f"Universal: "
        f"{universal_detection_rate:.2f}%"
    )

    print()

    # ----------------------------------------------
    # Confidence reductions
    # ----------------------------------------------

    random_drop = (
        average_clean
        -
        average_random
    )

    universal_drop = (
        average_clean
        -
        average_universal
    )

    print(
        "AVERAGE CONFIDENCE DROP"
    )

    print(
        f"Random patch:    "
        f"{random_drop:.4f}"
    )

    print(
        f"Universal patch: "
        f"{universal_drop:.4f}"
    )

    print()

    # ----------------------------------------------
    # Interpretation
    # ----------------------------------------------

    if (
        average_universal
        <
        average_random
    ):

        print(
            "SUCCESS:"
        )

        print(
            "The universal patch reduced "
            "average person confidence more "
            "than the random patch."
        )

    else:

        print(
            "NOTE:"
        )

        print(
            "The universal patch did not "
            "outperform the random patch "
            "on the validation set."
        )

    print()

    if (
        universal_detection_rate
        <
        clean_detection_rate
    ):

        print(
            "The universal patch also "
            "reduced the person detection rate."
        )

    else:

        print(
            "The universal patch did not "
            "reduce the detection rate."
        )


if __name__ == "__main__":
    main()