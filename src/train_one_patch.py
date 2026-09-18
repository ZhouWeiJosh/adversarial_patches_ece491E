from pathlib import Path

import torch
import torch.nn.functional as F

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

IMAGE_PATH = (
    ROOT
    / "data"
    / "test_images"
    / "crop001001.png"
)

PATCH_OUTPUT = (
    ROOT
    / "patches"
    / "one_image_patch.png"
)

PATCHED_OUTPUT = (
    ROOT
    / "results"
    / "images"
    / "trained_patch_result.png"
)


# ==================================================
# Settings
# ==================================================

IMAGE_SIZE = 416

PATCH_SIZE = 125

NUM_STEPS = 2000

LEARNING_RATE = 0.03

ATTACK_TYPE = "obj"

CONF_THRESHOLD = 0.4
NMS_THRESHOLD = 0.4

# Where vertically on the person to place the patch.
# 0.0 = top of bounding box
# 0.5 = center
TORSO_POSITION = 0.40

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ==================================================
# Patch overlay
# ==================================================

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
        YOLO normalized bbox:
        [center_x, center_y, width, height, ...]
    """

    patched = image.clone()

    _, _, image_h, image_w = patched.shape

    # YOLO box values are normalized 0-1
    person_center_x = (
        person_box[0] * image_w
    )

    person_center_y = (
        person_box[1] * image_h
    )

    person_width = (
        person_box[2] * image_w
    )

    person_height = (
        person_box[3] * image_h
    )

    # Top of person's bounding box
    person_top = (
        person_center_y
        -
        person_height / 2
    )

    # Put patch around upper torso
    patch_center_x = person_center_x

    patch_center_y = (
        person_top
        +
        TORSO_POSITION
        * person_height
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

    # Clamp to image boundaries
    x1 = max(0, x1)
    y1 = max(0, y1)

    x2 = min(image_w, x2)
    y2 = min(image_h, y2)

    patch_width = x2 - x1
    patch_height = y2 - y1

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

# ==================================================
# Main
# ==================================================

def main():

    print("======================================")
    print("TRAIN ONE ADVERSARIAL PATCH")
    print("======================================")

    print(f"Device: {DEVICE}")
    print(f"Attack: {ATTACK_TYPE}")
    print()

    # ----------------------------------------------
    # YOLO
    # ----------------------------------------------

    print("Loading YOLOv2...")

    model = Darknet(
        str(CFG_PATH)
    )

    model.load_weights(
        str(WEIGHTS_PATH)
    )

    model = model.to(DEVICE)

    model.eval()

    for parameter in model.parameters():

        parameter.requires_grad = False

    print("YOLO loaded.")

    # ----------------------------------------------
    # Image
    # ----------------------------------------------

    image = Image.open(
        IMAGE_PATH
    ).convert("RGB")

    image = image.resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        )
    )

    print("Finding person bounding box...")

    boxes = do_detect(
        model,
        image,
        CONF_THRESHOLD,
        NMS_THRESHOLD,
        DEVICE.type == "cuda"
    )

    person_boxes = []

    for box in boxes:

        class_id = int(box[6])

        # COCO person = class 0
        if class_id == 0:
            person_boxes.append(box)


    if len(person_boxes) == 0:

        raise RuntimeError(
            "No person detected in training image."
        )


    # Use the person with highest detection confidence
    person_box = max(
        person_boxes,
        key=lambda box:
            float(
                box[4].detach().cpu()
                if torch.is_tensor(box[4])
                else box[4]
            )
            *
            float(
                box[5].detach().cpu()
                if torch.is_tensor(box[5])
                else box[5]
            )
    )


    print("Person found.")

    print(
        "Bounding box:",
        person_box[:4]
    )

    image_tensor = (
        TF.to_tensor(image)
        .unsqueeze(0)
        .to(DEVICE)
    )

    # ----------------------------------------------
    # Patch
    # ----------------------------------------------

    patch = torch.rand(
        3,
        PATCH_SIZE,
        PATCH_SIZE,
        device=DEVICE,
        requires_grad=True
    )

    # ----------------------------------------------
    # Optimizer
    # ----------------------------------------------

    optimizer = torch.optim.Adam(
        [patch],
        lr=LEARNING_RATE
    )

    # ----------------------------------------------
    # YOLO loss
    # ----------------------------------------------

    attack_loss_function = (
        YOLOv2PersonLoss(
            attack_type=ATTACK_TYPE
        )
    )

    # ----------------------------------------------
    # Initial score
    # ----------------------------------------------

    with torch.no_grad():

        clean_output = model(
            image_tensor
        )

        clean_score = (
            attack_loss_function(
                clean_output
            )
        )

    print()
    print(
        f"Initial clean attack score: "
        f"{clean_score.item():.6f}"
    )

    print()
    print("Starting optimization...")
    print()

    # ==============================================
    # TRAINING LOOP
    # ==============================================

    for step in range(
        1,
        NUM_STEPS + 1
    ):

        optimizer.zero_grad()

        # ------------------------------------------
        # Keep patch valid
        # ------------------------------------------

        patch_clamped = torch.clamp(
            patch,
            0.0,
            1.0
        )

        # ------------------------------------------
        # Add patch to image
        # ------------------------------------------

        patched_image = apply_patch_on_person(
            image_tensor,
            patch_clamped,
            person_box
        )

        # ------------------------------------------
        # YOLO forward pass
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

        # We MINIMIZE this score
        total_loss = attack_loss

        # ------------------------------------------
        # Backward
        # ------------------------------------------

        total_loss.backward()

        optimizer.step()

        # ------------------------------------------
        # Force legal RGB range
        # ------------------------------------------

        with torch.no_grad():

            patch.clamp_(
                0.0,
                1.0
            )

        # ------------------------------------------
        # Display progress
        # ------------------------------------------

        if (
            step == 1
            or
            step % 25 == 0
        ):

            print(
                f"Step "
                f"{step:4d}/{NUM_STEPS}"
                f" | loss: "
                f"{total_loss.item():.6f}"
            )

    # ==============================================
    # SAVE PATCH
    # ==============================================

    PATCH_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    patch_image = (
        TF.to_pil_image(
            patch.detach().cpu()
        )
    )

    patch_image.save(
        PATCH_OUTPUT
    )

    # ==============================================
    # SAVE PATCHED IMAGE
    # ==============================================

    final_patch = torch.clamp(
        patch,
        0.0,
        1.0
    )

    final_image = apply_patch_on_person(
        image_tensor,
        final_patch,
        person_box
    )
    
    final_pil = TF.to_pil_image(
        final_image[
            0
        ].detach().cpu()
    )

    PATCHED_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    final_pil.save(
        PATCHED_OUTPUT
    )

    # ==============================================
    # FINAL SCORE
    # ==============================================

    with torch.no_grad():

        final_output = model(
            final_image
        )

        final_score = (
            attack_loss_function(
                final_output
            )
        )

    print()
    print("======================================")
    print("TRAINING COMPLETE")
    print("======================================")

    print(
        f"Initial score: "
        f"{clean_score.item():.6f}"
    )

    print(
        f"Final score:   "
        f"{final_score.item():.6f}"
    )

    print()
    print(
        f"Patch saved to:\n"
        f"{PATCH_OUTPUT}"
    )

    print()
    print(
        f"Patched image saved to:\n"
        f"{PATCHED_OUTPUT}"
    )


if __name__ == "__main__":
    main()
