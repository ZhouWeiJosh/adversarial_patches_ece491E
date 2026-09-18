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
from pathlib import Path
import random
import math

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

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

# If you cloned the original adversarial-yolo repository,
# this file should usually exist.
PRINTABILITY_FILE = (
    ROOT
    / "non_printability"
    / "30values.txt"
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


# ==================================================
# TV + NPS weights
# ==================================================

# Paper:
#
# L = attack_loss + alpha * NPS + beta * TV
#
# The paper says alpha and beta were selected empirically,
# so these are adjustable starting values.

NPS_WEIGHT = 0.01

TV_WEIGHT = 2.5


# ==================================================
# Random transformation settings
# ==================================================

# Paper explicitly says +/- 20 degrees.
MAX_ROTATION = 20.0

# Paper says randomly scale up/down,
# but does not specify exact percentages.
MIN_SCALE = 0.80
MAX_SCALE = 1.20

# Brightness multiplier
MIN_BRIGHTNESS = 0.80
MAX_BRIGHTNESS = 1.20

# Contrast multiplier
MIN_CONTRAST = 0.80
MAX_CONTRAST = 1.20

# Gaussian noise standard deviation
MAX_NOISE = 0.10


# Start small while debugging.
# Change to None when ready to use everything.
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


# ==================================================
# Total Variation Loss
# ==================================================

def total_variation_loss(patch):
    """
    Total Variation loss.

    Encourages neighboring pixels to have similar colors.

    patch shape:
        [3, H, W]
    """

    # Difference between vertically neighboring pixels
    vertical_difference = (
        patch[:, 1:, :]
        -
        patch[:, :-1, :]
    )

    # Difference between horizontally neighboring pixels
    horizontal_difference = (
        patch[:, :, 1:]
        -
        patch[:, :, :-1]
    )

    vertical_loss = (
        vertical_difference
        .pow(2)
        .mean()
    )

    horizontal_loss = (
        horizontal_difference
        .pow(2)
        .mean()
    )

    tv_loss = torch.sqrt(
        vertical_loss
        +
        horizontal_loss
        +
        1e-8
    )

    return tv_loss


# ==================================================
# NPS
# ==================================================

def load_printable_colors():
    """
    Load printable RGB values from:

        non_printability/30values.txt

    Expected examples:

        0.0,0.0,0.0
        1.0,1.0,1.0

    Returns:

        Tensor [N, 3]
    """

    printable_colors = []

    if PRINTABILITY_FILE.exists():

        print(
            f"Loading printable colors from:"
        )

        print(
            PRINTABILITY_FILE
        )

        with open(
            PRINTABILITY_FILE,
            "r"
        ) as file:

            for line in file:

                line = line.strip()

                if not line:
                    continue

                # Handle either commas or spaces
                line = line.replace(
                    ",",
                    " "
                )

                values = line.split()

                if len(values) < 3:
                    continue

                try:

                    r = float(values[0])
                    g = float(values[1])
                    b = float(values[2])

                except ValueError:
                    continue

                printable_colors.append(
                    [
                        r,
                        g,
                        b
                    ]
                )

    # --------------------------------------------------
    # Fallback if 30values.txt is missing
    # --------------------------------------------------

    if len(printable_colors) == 0:

        print()

        print(
            "WARNING:"
        )

        print(
            "Could not load "
            "non_printability/30values.txt"
        )

        print(
            "Using a fallback printable-color palette."
        )

        print(
            "For a paper-faithful experiment, use the "
            "30values.txt file from adversarial-yolo."
        )

        print()

        printable_colors = [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.5],
            [0.0, 0.0, 1.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.5, 0.5],
            [0.0, 0.5, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.5],
            [0.0, 1.0, 1.0],

            [0.5, 0.0, 0.0],
            [0.5, 0.0, 0.5],
            [0.5, 0.0, 1.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 1.0],
            [0.5, 1.0, 0.0],
            [0.5, 1.0, 0.5],
            [0.5, 1.0, 1.0],

            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.5],
            [1.0, 0.0, 1.0],
            [1.0, 0.5, 0.0],
            [1.0, 0.5, 0.5],
            [1.0, 0.5, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.5],
            [1.0, 1.0, 1.0],

            [0.25, 0.25, 0.25],
            [0.75, 0.75, 0.75],
            [1.0, 0.75, 0.5],
        ]

    colors = torch.tensor(
        printable_colors,
        dtype=torch.float32,
        device=DEVICE
    )

    print(
        f"Printable colors loaded: "
        f"{len(colors)}"
    )

    print()

    return colors


def non_printability_loss(
    patch,
    printable_colors
):
    """
    Non-Printability Score.

    For every patch pixel:

        find the nearest printable RGB color

    Then average the distance.

    patch:
        [3, H, W]

    printable_colors:
        [N, 3]
    """

    # ----------------------------------------------
    # [3, H, W]
    # ->
    # [H * W, 3]
    # ----------------------------------------------

    pixels = (
        patch
        .permute(
            1,
            2,
            0
        )
        .reshape(
            -1,
            3
        )
    )

    # ----------------------------------------------
    # pixels:
    # [P, 1, 3]
    #
    # printable:
    # [1, C, 3]
    #
    # result:
    # [P, C, 3]
    # ----------------------------------------------

    difference = (
        pixels.unsqueeze(1)
        -
        printable_colors.unsqueeze(0)
    )

    # Euclidean RGB distance
    distance = torch.sqrt(
        torch.sum(
            difference.pow(2),
            dim=2
        )
        +
        1e-8
    )

    # For each pixel,
    # find nearest printable color
    minimum_distance = torch.min(
        distance,
        dim=1
    ).values

    nps_loss = (
        minimum_distance.mean()
    )

    return nps_loss


# ==================================================
# Random Patch Transformations
# ==================================================

def random_transform_patch(patch):
    """
    Apply transformations described in the paper:

        - rotation
        - scale
        - random noise
        - brightness
        - contrast

    Everything here remains differentiable with
    respect to the patch.

    Returns:

        transformed_patch
        transformed_mask
    """

    # ----------------------------------------------
    # Start with patch
    # ----------------------------------------------

    transformed_patch = patch

    # Mask tells us which pixels really belong
    # to the rotated patch.

    mask = torch.ones(
        1,
        patch.shape[1],
        patch.shape[2],
        device=patch.device,
        dtype=patch.dtype
    )

    # ==================================================
    # Brightness
    # ==================================================

    brightness_factor = random.uniform(
        MIN_BRIGHTNESS,
        MAX_BRIGHTNESS
    )

    transformed_patch = (
        transformed_patch
        *
        brightness_factor
    )


    # ==================================================
    # Contrast
    # ==================================================

    contrast_factor = random.uniform(
        MIN_CONTRAST,
        MAX_CONTRAST
    )

    mean_color = transformed_patch.mean(
        dim=(1, 2),
        keepdim=True
    )

    transformed_patch = (
        (
            transformed_patch
            -
            mean_color
        )
        *
        contrast_factor
        +
        mean_color
    )


    # ==================================================
    # Noise
    # ==================================================

    noise_strength = random.uniform(
        0.0,
        MAX_NOISE
    )

    noise = (
        torch.randn_like(
            transformed_patch
        )
        *
        noise_strength
    )

    transformed_patch = (
        transformed_patch
        +
        noise
    )


    # ==================================================
    # Scale
    # ==================================================

    scale_factor = random.uniform(
        MIN_SCALE,
        MAX_SCALE
    )

    original_height = (
        transformed_patch.shape[1]
    )

    original_width = (
        transformed_patch.shape[2]
    )

    new_height = max(
        1,
        int(
            original_height
            *
            scale_factor
        )
    )

    new_width = max(
        1,
        int(
            original_width
            *
            scale_factor
        )
    )

    transformed_patch = F.interpolate(
        transformed_patch.unsqueeze(0),
        size=(
            new_height,
            new_width
        ),
        mode="bilinear",
        align_corners=False
    ).squeeze(0)

    mask = F.interpolate(
        mask.unsqueeze(0),
        size=(
            new_height,
            new_width
        ),
        mode="bilinear",
        align_corners=False
    ).squeeze(0)


    # ==================================================
    # Rotation
    # ==================================================

    angle = random.uniform(
        -MAX_ROTATION,
        MAX_ROTATION
    )

    transformed_patch = TF.rotate(
        transformed_patch,
        angle=angle,
        interpolation=InterpolationMode.BILINEAR,
        expand=True,
        fill=0.0
    )

    mask = TF.rotate(
        mask,
        angle=angle,
        interpolation=InterpolationMode.BILINEAR,
        expand=True,
        fill=0.0
    )


    # ==================================================
    # Keep valid image values
    # ==================================================

    transformed_patch = torch.clamp(
        transformed_patch,
        0.0,
        1.0
    )

    mask = torch.clamp(
        mask,
        0.0,
        1.0
    )

    return (
        transformed_patch,
        mask
    )


# ==================================================
# Apply Patch
# ==================================================

def apply_patch_on_person(
    image,
    patch,
    patch_mask,
    person_box
):
    """
    image:
        [1, 3, H, W]

    patch:
        [3, P_H, P_W]

    patch_mask:
        [1, P_H, P_W]

    person_box:
        normalized YOLO bbox

        [
            center_x,
            center_y,
            width,
            height,
            ...
        ]
    """

    patched = image.clone()

    _, _, image_h, image_w = (
        patched.shape
    )

    # ----------------------------------------------
    # Person bounding box
    # ----------------------------------------------

    person_center_x = (
        get_box_value(
            person_box[0]
        )
        *
        image_w
    )

    person_center_y = (
        get_box_value(
            person_box[1]
        )
        *
        image_h
    )

    person_height = (
        get_box_value(
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

    x2 = (
        x1
        +
        patch_w
    )

    y2 = (
        y1
        +
        patch_h
    )


    # ==================================================
    # Figure out valid overlap
    # ==================================================

    image_x1 = max(
        0,
        x1
    )

    image_y1 = max(
        0,
        y1
    )

    image_x2 = min(
        image_w,
        x2
    )

    image_y2 = min(
        image_h,
        y2
    )


    # Patch completely outside image
    if (
        image_x1 >= image_x2
        or
        image_y1 >= image_y2
    ):

        return patched


    # ----------------------------------------------
    # Corresponding patch crop
    # ----------------------------------------------

    patch_x1 = (
        image_x1
        -
        x1
    )

    patch_y1 = (
        image_y1
        -
        y1
    )

    patch_x2 = (
        patch_x1
        +
        (
            image_x2
            -
            image_x1
        )
    )

    patch_y2 = (
        patch_y1
        +
        (
            image_y2
            -
            image_y1
        )
    )


    # ----------------------------------------------
    # Crop patch
    # ----------------------------------------------

    patch_crop = patch[
        :,
        patch_y1:patch_y2,
        patch_x1:patch_x2
    ].unsqueeze(0)

    mask_crop = patch_mask[
        :,
        patch_y1:patch_y2,
        patch_x1:patch_x2
    ].unsqueeze(0)


    # ----------------------------------------------
    # Current image region
    # ----------------------------------------------

    image_region = patched[
        :,
        :,
        image_y1:image_y2,
        image_x1:image_x2
    ]


    # ----------------------------------------------
    # Alpha blend
    # ----------------------------------------------

    patched_region = (
        image_region
        *
        (
            1.0
            -
            mask_crop
        )
        +
        patch_crop
        *
        mask_crop
    )


    # ----------------------------------------------
    # Put it back
    # ----------------------------------------------

    patched[
        :,
        :,
        image_y1:image_y2,
        image_x1:image_x2
    ] = patched_region

    return patched


# ==================================================
# Find Images
# ==================================================

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
            directory.glob(
                extension
            )
        )

    return sorted(
        images
    )


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
        "TV + NPS + RANDOM TRANSFORMATIONS"
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

    print(
        f"NPS weight: {NPS_WEIGHT}"
    )

    print(
        f"TV weight:  {TV_WEIGHT}"
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
            "Training directory does not exist:"
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
    # Limit images while debugging
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
    # Printable colors
    # ==================================================

    printable_colors = (
        load_printable_colors()
    )


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


        epoch_total_loss = 0.0
        epoch_attack_loss = 0.0
        epoch_nps_loss = 0.0
        epoch_tv_loss = 0.0

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
            # Convert image to Tensor
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


            # ------------------------------------------
            # Clamp original patch
            # ------------------------------------------

            patch_clamped = torch.clamp(
                patch,
                0.0,
                1.0
            )


            # ==========================================
            # TV LOSS
            # ==========================================

            tv_loss = total_variation_loss(
                patch_clamped
            )


            # ==========================================
            # NPS LOSS
            # ==========================================

            nps_loss = non_printability_loss(
                patch_clamped,
                printable_colors
            )


            # ==========================================
            # Random transformations
            # ==========================================

            (
                transformed_patch,
                transformed_mask

            ) = random_transform_patch(
                patch_clamped
            )


            # ==========================================
            # Apply transformed patch
            # ==========================================

            patched_image = (
                apply_patch_on_person(
                    image_tensor,
                    transformed_patch,
                    transformed_mask,
                    person_box
                )
            )


            # ==========================================
            # YOLO forward
            # ==========================================

            output = model(
                patched_image
            )


            # ==========================================
            # Attack Loss
            # ==========================================

            attack_loss = (
                attack_loss_function(
                    output
                )
            )


            # ==========================================
            # Total Loss
            # ==========================================

            total_loss = (
                attack_loss
                +
                NPS_WEIGHT
                *
                nps_loss
                +
                TV_WEIGHT
                *
                tv_loss
            )


            # ==========================================
            # Backprop
            # ==========================================

            total_loss.backward()

            optimizer.step()


            # ------------------------------------------
            # Keep patch RGB legal
            # ------------------------------------------

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


            epoch_total_loss += (
                total_loss.item()
            )

            epoch_attack_loss += (
                attack_loss.item()
            )

            epoch_nps_loss += (
                nps_loss.item()
            )

            epoch_tv_loss += (
                tv_loss.item()
            )


            print(
                f"[{image_number:3d}/"
                f"{len(training_images):3d}] "
                f"{image_path.name} "
                f"| total: "
                f"{total_loss.item():.6f} "
                f"| attack: "
                f"{attack_loss.item():.6f} "
                f"| nps: "
                f"{nps_loss.item():.6f} "
                f"| tv: "
                f"{tv_loss.item():.6f}"
            )


        # ==================================================
        # Epoch summary
        # ==================================================

        if images_used > 0:

            average_total = (
                epoch_total_loss
                /
                images_used
            )

            average_attack = (
                epoch_attack_loss
                /
                images_used
            )

            average_nps = (
                epoch_nps_loss
                /
                images_used
            )

            average_tv = (
                epoch_tv_loss
                /
                images_used
            )

        else:

            average_total = 0.0
            average_attack = 0.0
            average_nps = 0.0
            average_tv = 0.0


        print()

        print(
            f"Epoch {epoch} complete"
        )

        print(
            f"Images used:     "
            f"{images_used}"
        )

        print(
            f"Images skipped:  "
            f"{images_skipped}"
        )

        print(
            f"Average TOTAL:   "
            f"{average_total:.6f}"
        )

        print(
            f"Average ATTACK:  "
            f"{average_attack:.6f}"
        )

        print(
            f"Average NPS:     "
            f"{average_nps:.6f}"
        )

        print(
            f"Average TV:      "
            f"{average_tv:.6f}"
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
                .clamp(
                    0.0,
                    1.0
                )
                .cpu()
            )
        )


        patch_image.save(
            PATCH_OUTPUT
        )


        print(
            "Patch saved:"
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
        "Universal patch:"
    )

    print(
        PATCH_OUTPUT
    )


if __name__ == "__main__":
    main()