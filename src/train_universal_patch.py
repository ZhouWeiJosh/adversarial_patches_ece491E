from pathlib import Path
import random

import torch
from PIL import Image
from torchvision.transforms import functional as TF

from darknet import Darknet
from utils import do_detect
from yolo_loss import YOLOv2PersonLoss


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

TRAIN_DIR = (
    ROOT
    / "data"
    / "train_images"
)

PATCH_OUTPUT = (
    ROOT
    / "patches"
    / "universal_patch.png"
)


# ==================================================
# Settings
# ==================================================

IMAGE_SIZE = 416

PATCH_SIZE = 125

NUM_EPOCHS = 5

LEARNING_RATE = 0.03

ATTACK_TYPE = "obj"

CONF_THRESHOLD = 0.4
NMS_THRESHOLD = 0.4

TORSO_POSITION = 0.40

# Start small while debugging.
# Set to None later to use every image.
MAX_IMAGES = 50


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ==================================================
# Helpers
# ==================================================

def get_box_value(value):
    """
    Convert YOLO tensor values into normal Python floats.
    """

    if torch.is_tensor(value):
        return (
            value
            .detach()
            .cpu()
            .item()
        )

    return float(value)


def get_best_person_box(boxes):
    """
    Find the highest-confidence person detection.

    COCO class 0 = person.
    """

    person_boxes = []

    for box in boxes:

        class_id = int(box[6])

        if class_id == 0:
            person_boxes.append(box)

    if len(person_boxes) == 0:
        return None

    def confidence(box):

        objectness = get_box_value(
            box[4]
        )

        class_confidence = get_box_value(
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


def apply_patch_on_person(
    image,
    patch,
    person_box
):
    """
    image:
        [1, 3, H, W]

    patch:
        [3, P, P]

    person_box:
        normalized YOLO bbox
        [center_x, center_y, width, height, ...]
    """

    patched = image.clone()

    _, _, image_h, image_w = (
        patched.shape
    )

    # ----------------------------------------------
    # Person bounding box
    # ----------------------------------------------

    person_center_x = (
        get_box_value(person_box[0])
        *
        image_w
    )

    person_center_y = (
        get_box_value(person_box[1])
        *
        image_h
    )

    person_height = (
        get_box_value(person_box[3])
        *
        image_h
    )

    # Top edge of the person's bbox
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
    # Keep patch inside the image
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
    # Place patch
    # ----------------------------------------------

    patched[
        :,
        :,
        y1:y2,
        x1:x2
    ] = patch[
        :,
        0:patch_height,
        0:patch_width
    ].unsqueeze(0)

    return patched


def find_training_images(directory):
    """
    Find common image formats.
    """

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
        "TRAIN UNIVERSAL ADVERSARIAL PATCH"
    )

    print(
        "======================================"
    )

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Attack: {ATTACK_TYPE}"
    )

    print()

    # ==================================================
    # Check dataset
    # ==================================================

    if not TRAIN_DIR.exists():

        print(
            "ERROR:"
        )

        print(
            "INRIA training directory "
            "does not exist:"
        )

        print(
            TRAIN_DIR
        )

        return

    training_images = (
        find_training_images(
            TRAIN_DIR
        )
    )

    if len(training_images) == 0:

        print(
            "ERROR:"
        )

        print(
            "No training images found."
        )

        return

    # ----------------------------------------------
    # Limit images during debugging
    # ----------------------------------------------

    if MAX_IMAGES is not None:

        training_images = (
            training_images[
                :MAX_IMAGES
            ]
        )

    print(
        f"Training images: "
        f"{len(training_images)}"
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

    model = model.to(
        DEVICE
    )

    model.eval()

    # Freeze YOLO
    for parameter in model.parameters():

        parameter.requires_grad = False

    print(
        "YOLO loaded."
    )

    print()

    # ==================================================
    # Create ONE universal patch
    # ==================================================

    patch = torch.rand(
        3,
        PATCH_SIZE,
        PATCH_SIZE,
        device=DEVICE,
        requires_grad=True
    )

    optimizer = torch.optim.Adam(
        [patch],
        lr=LEARNING_RATE
    )

    attack_loss_function = (
        YOLOv2PersonLoss(
            attack_type=ATTACK_TYPE
        )
    )

    # ==================================================
    # Training
    # ==================================================

    global_step = 0

    for epoch in range(
        1,
        NUM_EPOCHS + 1
    ):

        print(
            "======================================"
        )

        print(
            f"EPOCH {epoch}/{NUM_EPOCHS}"
        )

        print(
            "======================================"
        )

        # Different image order each epoch
        random.shuffle(
            training_images
        )

        epoch_loss = 0.0

        images_used = 0
        images_skipped = 0

        for image_number, image_path in enumerate(
            training_images,
            start=1
        ):

            # ==========================================
            # Load image
            # ==========================================

            try:

                image = Image.open(
                    image_path
                ).convert("RGB")

            except Exception as error:

                print(
                    f"Skipping "
                    f"{image_path.name}: "
                    f"{error}"
                )

                images_skipped += 1

                continue

            # Resize exactly like previous training
            image = image.resize(
                (
                    IMAGE_SIZE,
                    IMAGE_SIZE
                )
            )

            # ==========================================
            # Find person in CLEAN image
            # ==========================================

            boxes = do_detect(
                model,
                image,
                CONF_THRESHOLD,
                NMS_THRESHOLD,
                DEVICE.type == "cuda"
            )

            person_box = (
                get_best_person_box(
                    boxes
                )
            )

            if person_box is None:

                print(
                    f"[{image_number}/"
                    f"{len(training_images)}] "
                    f"{image_path.name}: "
                    f"no person detected - skipped"
                )

                images_skipped += 1

                continue

            # ==========================================
            # Convert image to tensor
            # ==========================================

            image_tensor = (
                TF.to_tensor(
                    image
                )
                .unsqueeze(0)
                .to(DEVICE)
            )

            # ==========================================
            # Training step
            # ==========================================

            optimizer.zero_grad()

            patch_clamped = (
                torch.clamp(
                    patch,
                    0.0,
                    1.0
                )
            )

            patched_image = (
                apply_patch_on_person(
                    image_tensor,
                    patch_clamped,
                    person_box
                )
            )

            # ------------------------------------------
            # YOLO forward
            # ------------------------------------------

            output = model(
                patched_image
            )

            # ------------------------------------------
            # Attack loss
            # ------------------------------------------

            attack_loss = (
                attack_loss_function(
                    output
                )
            )

            total_loss = (
                attack_loss
            )

            # ------------------------------------------
            # Backprop
            # ------------------------------------------

            total_loss.backward()

            optimizer.step()

            # Keep RGB values legal
            with torch.no_grad():

                patch.clamp_(
                    0.0,
                    1.0
                )

            # ==========================================
            # Statistics
            # ==========================================

            global_step += 1

            images_used += 1

            epoch_loss += (
                total_loss.item()
            )

            print(
                f"[{image_number:3d}/"
                f"{len(training_images):3d}] "
                f"{image_path.name} "
                f"| loss: "
                f"{total_loss.item():.6f}"
            )

        # ==================================================
        # Epoch summary
        # ==================================================

        if images_used > 0:

            average_loss = (
                epoch_loss
                /
                images_used
            )

        else:

            average_loss = 0.0

        print()

        print(
            f"Epoch {epoch} complete"
        )

        print(
            f"Images used:    "
            f"{images_used}"
        )

        print(
            f"Images skipped: "
            f"{images_skipped}"
        )

        print(
            f"Average loss:   "
            f"{average_loss:.6f}"
        )

        print()

        # ==================================================
        # Save after every epoch
        # ==================================================

        PATCH_OUTPUT.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        patch_image = (
            TF.to_pil_image(
                patch
                .detach()
                .cpu()
            )
        )

        patch_image.save(
            PATCH_OUTPUT
        )

        print(
            f"Patch saved:"
        )

        print(
            PATCH_OUTPUT
        )

        print()

    # ==================================================
    # Complete
    # ==================================================

    print(
        "======================================"
    )

    print(
        "UNIVERSAL TRAINING COMPLETE"
    )

    print(
        "======================================"
    )

    print(
        f"Total updates: "
        f"{global_step}"
    )

    print()

    print(
        f"Universal patch:"
    )

    print(
        PATCH_OUTPUT
    )


if __name__ == "__main__":
    main()
