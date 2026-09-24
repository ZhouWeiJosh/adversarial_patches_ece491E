from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

from darknet import Darknet
from utils import do_detect
from patch_utils import create_random_patch


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

CFG_PATH = ROOT / "cfg" / "yolo.cfg"
WEIGHTS_PATH = ROOT / "weights" / "yolo.weights"
DATA_DIR = ROOT / "data" / "validation_images"

UNIVERSAL_PATCH_PATH = ROOT / "patches" / "universal_patch_obj.png"

OUTPUT_DIR = ROOT / "results"
OUTPUT_PLOT = OUTPUT_DIR / "pr_curve.png"
OUTPUT_CSV = OUTPUT_DIR / "pr_curve.csv"


# ============================================================
# SETTINGS
# ============================================================

YOLO_SIZE = 416
PATCH_SIZE = 125
TORSO_POSITION = 0.40

# IMPORTANT:
# This must be LOW for a PR curve. If you use 0.4 here, all detections
# below 0.4 disappear before the PR curve can be calculated.
DETECTION_FLOOR = 0.001

NMS_THRESHOLD = 0.4

# IoU used to decide whether a predicted person box matches the
# pseudo-ground-truth clean detection.
IOU_THRESHOLD = 0.5

NUM_VALIDATION_IMAGES = None  # None = all images

# Number of random transformed evaluations per image.
# 1 is fast. Increase to 3-5 for a smoother EOT-style result.
TRANSFORM_REPEATS = 1

USE_RANDOM_TRANSFORMS = True

MAX_ROTATION = 20.0
MIN_SCALE = 0.80
MAX_SCALE = 1.20
MIN_BRIGHTNESS = 0.80
MAX_BRIGHTNESS = 1.20
MIN_CONTRAST = 0.80
MAX_CONTRAST = 1.20
MAX_NOISE = 0.10

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
USE_CUDA = DEVICE.type == "cuda"


# ============================================================
# HELPERS
# ============================================================

def get_value(value):
    if torch.is_tensor(value):
        return value.detach().cpu().item()
    return float(value)


def person_confidence(box):
    """
    utils.do_detect() boxes are expected to contain:
      [cx, cy, w, h, objectness, class_confidence, class_id]
    """
    return get_value(box[4]) * get_value(box[5])


def get_person_boxes(boxes):
    return [box for box in boxes if int(box[6]) == 0]


def yolo_box_to_xyxy(box):
    """
    Convert normalized YOLO [cx,cy,w,h,...] to normalized [x1,y1,x2,y2].
    """
    cx = get_value(box[0])
    cy = get_value(box[1])
    w = get_value(box[2])
    h = get_value(box[3])

    return (
        cx - w / 2.0,
        cy - h / 2.0,
        cx + w / 2.0,
        cy + h / 2.0,
    )


def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = yolo_box_to_xyxy(box_a)
    bx1, by1, bx2, by2 = yolo_box_to_xyxy(box_b)

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter_area

    if union <= 0.0:
        return 0.0

    return inter_area / union


def find_images(directory):
    extensions = [
        "*.png", "*.jpg", "*.jpeg",
        "*.PNG", "*.JPG", "*.JPEG",
    ]

    result = []
    for extension in extensions:
        result.extend(directory.glob(extension))

    return sorted(result)


def run_detection(model, image):
    return do_detect(
        model,
        image,
        DETECTION_FLOOR,
        NMS_THRESHOLD,
        USE_CUDA,
    )


def load_patch(path):
    image = Image.open(path).convert("RGB")
    image = image.resize((PATCH_SIZE, PATCH_SIZE))
    return TF.to_tensor(image)


# ============================================================
# PATCH TRANSFORM
# ============================================================

def random_transform_patch(patch):
    if not USE_RANDOM_TRANSFORMS:
        mask = torch.ones(
            1,
            patch.shape[1],
            patch.shape[2],
            dtype=patch.dtype,
        )
        return patch.clone(), mask

    transformed = patch.clone()

    brightness = random.uniform(
        MIN_BRIGHTNESS,
        MAX_BRIGHTNESS,
    )
    transformed = transformed * brightness

    contrast = random.uniform(
        MIN_CONTRAST,
        MAX_CONTRAST,
    )
    mean = transformed.mean(dim=(1, 2), keepdim=True)
    transformed = (transformed - mean) * contrast + mean

    noise_strength = random.uniform(0.0, MAX_NOISE)
    transformed = transformed + torch.randn_like(transformed) * noise_strength

    scale = random.uniform(MIN_SCALE, MAX_SCALE)

    new_h = max(1, int(transformed.shape[1] * scale))
    new_w = max(1, int(transformed.shape[2] * scale))

    transformed = torch.nn.functional.interpolate(
        transformed.unsqueeze(0),
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    mask = torch.ones(
        1,
        patch.shape[1],
        patch.shape[2],
        dtype=patch.dtype,
    )

    mask = torch.nn.functional.interpolate(
        mask.unsqueeze(0),
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    angle = random.uniform(-MAX_ROTATION, MAX_ROTATION)

    transformed = TF.rotate(
        transformed,
        angle=angle,
        interpolation=InterpolationMode.BILINEAR,
        expand=True,
        fill=0.0,
    )

    mask = TF.rotate(
        mask,
        angle=angle,
        interpolation=InterpolationMode.BILINEAR,
        expand=True,
        fill=0.0,
    )

    transformed = transformed.clamp(0.0, 1.0)
    mask = mask.clamp(0.0, 1.0)

    return transformed, mask


def apply_patch_on_person(image_tensor, patch, patch_mask, person_box):
    """
    image_tensor: [3,H,W]
    patch:        [3,PH,PW]
    patch_mask:   [1,PH,PW]
    """

    patched = image_tensor.clone()

    _, image_h, image_w = patched.shape

    person_center_x = get_value(person_box[0]) * image_w
    person_center_y = get_value(person_box[1]) * image_h
    person_height = get_value(person_box[3]) * image_h

    person_top = person_center_y - person_height / 2.0

    patch_center_x = person_center_x
    patch_center_y = person_top + TORSO_POSITION * person_height

    patch_h = patch.shape[1]
    patch_w = patch.shape[2]

    x1 = int(patch_center_x - patch_w / 2)
    y1 = int(patch_center_y - patch_h / 2)
    x2 = x1 + patch_w
    y2 = y1 + patch_h

    image_x1 = max(0, x1)
    image_y1 = max(0, y1)
    image_x2 = min(image_w, x2)
    image_y2 = min(image_h, y2)

    if image_x1 >= image_x2 or image_y1 >= image_y2:
        return patched

    patch_x1 = image_x1 - x1
    patch_y1 = image_y1 - y1
    patch_x2 = patch_x1 + (image_x2 - image_x1)
    patch_y2 = patch_y1 + (image_y2 - image_y1)

    patch_crop = patch[
        :,
        patch_y1:patch_y2,
        patch_x1:patch_x2,
    ]

    mask_crop = patch_mask[
        :,
        patch_y1:patch_y2,
        patch_x1:patch_x2,
    ]

    image_region = patched[
        :,
        image_y1:image_y2,
        image_x1:image_x2,
    ]

    patched[
        :,
        image_y1:image_y2,
        image_x1:image_x2,
    ] = (
        image_region * (1.0 - mask_crop)
        + patch_crop * mask_crop
    )

    return patched


def apply_patch_to_all_people(image_tensor, patch, clean_person_boxes):
    """
    Apply one transformed version of the patch to every clean person box.
    """
    patched = image_tensor.clone()

    for person_box in clean_person_boxes:
        transformed_patch, transformed_mask = random_transform_patch(patch)

        patched = apply_patch_on_person(
            patched,
            transformed_patch,
            transformed_mask,
            person_box,
        )

    return patched


# ============================================================
# PR EVALUATION
# ============================================================

def collect_predictions_for_condition(
    model,
    images,
    condition_name,
    patch=None,
):
    """
    Pseudo-ground truth:
      person boxes produced by YOLO on the CLEAN image.

    This follows the paper's setup closely enough to reproduce its
    evaluation idea without requiring separate annotation parsing.

    Returns:
      predictions: list of (confidence, is_true_positive)
      total_ground_truth: number of clean pseudo-GT person boxes
    """

    predictions = []
    total_ground_truth = 0
    used_images = 0
    skipped_images = 0

    for index, image_path in enumerate(images, start=1):
        try:
            clean_image = Image.open(image_path).convert("RGB")
        except Exception as error:
            print(f"[{condition_name}] skip {image_path.name}: {error}")
            skipped_images += 1
            continue

        clean_image = clean_image.resize((YOLO_SIZE, YOLO_SIZE))

        # ----------------------------------------------------
        # Generate pseudo-ground truth ONCE from clean image.
        # ----------------------------------------------------
        clean_boxes = run_detection(model, clean_image)
        gt_boxes = get_person_boxes(clean_boxes)

        if len(gt_boxes) == 0:
            skipped_images += 1
            print(
                f"[{condition_name}] "
                f"{index}/{len(images)} {image_path.name}: "
                f"no clean person -> skipped"
            )
            continue

        used_images += 1

        # Do not multiply GT count by transform repeats.
        # Each repeat is treated as another evaluation sample,
        # so we DO multiply it here.
        total_ground_truth += len(gt_boxes) * TRANSFORM_REPEATS

        clean_tensor = TF.to_tensor(clean_image)

        for repeat in range(TRANSFORM_REPEATS):

            if condition_name == "CLEAN":
                eval_image = clean_image

            else:
                if patch is None:
                    raise ValueError(
                        f"{condition_name} requires a patch"
                    )

                patched_tensor = apply_patch_to_all_people(
                    clean_tensor,
                    patch,
                    gt_boxes,
                )

                eval_image = TF.to_pil_image(
                    patched_tensor.clamp(0.0, 1.0)
                )

            predicted_boxes = get_person_boxes(
                run_detection(model, eval_image)
            )

            predicted_boxes = sorted(
                predicted_boxes,
                key=person_confidence,
                reverse=True,
            )

            # Greedy IoU matching.
            matched_gt = set()

            for prediction in predicted_boxes:
                best_iou = 0.0
                best_gt_index = None

                for gt_index, gt_box in enumerate(gt_boxes):
                    if gt_index in matched_gt:
                        continue

                    iou = box_iou(prediction, gt_box)

                    if iou > best_iou:
                        best_iou = iou
                        best_gt_index = gt_index

                is_true_positive = (
                    best_gt_index is not None
                    and best_iou >= IOU_THRESHOLD
                )

                if is_true_positive:
                    matched_gt.add(best_gt_index)

                predictions.append(
                    (
                        person_confidence(prediction),
                        int(is_true_positive),
                    )
                )

        print(
            f"[{condition_name}] "
            f"{index}/{len(images)} {image_path.name} "
            f"| clean persons: {len(gt_boxes)}"
        )

    print(
        f"\n{condition_name}: "
        f"used={used_images}, skipped={skipped_images}, "
        f"GT={total_ground_truth}, predictions={len(predictions)}\n"
    )

    return predictions, total_ground_truth


def precision_recall_curve_from_predictions(
    predictions,
    total_ground_truth,
):
    """
    Sort all detections by descending confidence and progressively
    include them. This is the standard object-detection PR construction.
    """

    if total_ground_truth == 0:
        return (
            np.array([0.0]),
            np.array([1.0]),
            np.array([1.0]),
            0.0,
        )

    if len(predictions) == 0:
        return (
            np.array([0.0]),
            np.array([1.0]),
            np.array([1.0]),
            0.0,
        )

    predictions = sorted(
        predictions,
        key=lambda item: item[0],
        reverse=True,
    )

    scores = np.array(
        [item[0] for item in predictions],
        dtype=np.float64,
    )

    tp = np.array(
        [item[1] for item in predictions],
        dtype=np.float64,
    )

    fp = 1.0 - tp

    cumulative_tp = np.cumsum(tp)
    cumulative_fp = np.cumsum(fp)

    recall = cumulative_tp / float(total_ground_truth)

    precision = cumulative_tp / np.maximum(
        cumulative_tp + cumulative_fp,
        1e-12,
    )

    # Precision envelope and area under PR curve.
    recall_for_ap = np.concatenate(([0.0], recall, [1.0]))
    precision_for_ap = np.concatenate(([0.0], precision, [0.0]))

    for i in range(len(precision_for_ap) - 2, -1, -1):
        precision_for_ap[i] = max(
            precision_for_ap[i],
            precision_for_ap[i + 1],
        )

    change_indices = np.where(
        recall_for_ap[1:] != recall_for_ap[:-1]
    )[0]

    ap = np.sum(
        (
            recall_for_ap[change_indices + 1]
            - recall_for_ap[change_indices]
        )
        * precision_for_ap[change_indices + 1]
    )

    return recall, precision, scores, ap


def recall_at_threshold(predictions, total_ground_truth, threshold):
    if total_ground_truth == 0:
        return 0.0

    # Since matching was performed before thresholding, count matched
    # predictions whose score survives the requested threshold.
    tp = sum(
        is_tp
        for score, is_tp in predictions
        if score >= threshold
    )

    return tp / total_ground_truth


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("PAPER-STYLE PERSON-DETECTION PR CURVE")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Detection floor: {DETECTION_FLOOR}")
    print(f"IoU threshold: {IOU_THRESHOLD}")
    print(f"Random transforms: {USE_RANDOM_TRANSFORMS}")
    print(f"Transform repeats: {TRANSFORM_REPEATS}")
    print()

    if not DATA_DIR.exists():
        print(f"ERROR: validation directory not found:\n{DATA_DIR}")
        return

    if not UNIVERSAL_PATCH_PATH.exists():
        print(f"ERROR: universal patch not found:\n{UNIVERSAL_PATCH_PATH}")
        return

    images = find_images(DATA_DIR)

    if NUM_VALIDATION_IMAGES is not None:
        images = images[:NUM_VALIDATION_IMAGES]

    if len(images) == 0:
        print("ERROR: no validation images found.")
        return

    print(f"Images selected: {len(images)}")
    print()

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("Loading YOLOv2...")

    model = Darknet(str(CFG_PATH))
    model.load_weights(str(WEIGHTS_PATH))
    model.eval()

    if USE_CUDA:
        model.cuda()

    print("YOLO loaded.\n")

    # --------------------------------------------------------
    # Load patches
    # --------------------------------------------------------

    universal_patch = load_patch(UNIVERSAL_PATCH_PATH)

    # Fixed random control patch for the whole test.
    # We still transform it independently on each application.
    random_patch = create_random_patch(
        patch_size=PATCH_SIZE
    ).detach().cpu()

    conditions = {
        "CLEAN": None,
        "NOISE": random_patch,
        "OBJ": universal_patch,
    }

    all_results = {}

    for name, patch in conditions.items():
        print("=" * 60)
        print(f"EVALUATING {name}")
        print("=" * 60)

        predictions, total_gt = collect_predictions_for_condition(
            model=model,
            images=images,
            condition_name=name,
            patch=patch,
        )

        recall, precision, scores, ap = (
            precision_recall_curve_from_predictions(
                predictions,
                total_gt,
            )
        )

        recall_04 = recall_at_threshold(
            predictions,
            total_gt,
            0.4,
        )

        all_results[name] = {
            "predictions": predictions,
            "gt": total_gt,
            "recall": recall,
            "precision": precision,
            "scores": scores,
            "ap": ap,
            "recall_04": recall_04,
        }

        print(
            f"{name}: AP={ap * 100:.2f}% "
            f"| recall@0.4={recall_04 * 100:.2f}%"
        )
        print()

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))

    for name, result in all_results.items():
        plt.plot(
            result["recall"],
            result["precision"],
            label=f"{name}: AP {result['ap'] * 100:.2f}%",
        )

    # Same visual working-point diagonal used in the paper.
    x = np.linspace(0.0, 1.0, 200)
    plt.plot(
        x,
        x,
        "--",
        linewidth=1.0,
        label="working-point diagonal",
    )

    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Person Detection Precision-Recall Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_PLOT,
        dpi=200,
    )

    print(f"Saved plot: {OUTPUT_PLOT}")

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    with open(OUTPUT_CSV, "w") as file:
        file.write(
            "condition,rank,confidence,precision,recall\n"
        )

        for name, result in all_results.items():
            recall = result["recall"]
            precision = result["precision"]
            scores = result["scores"]

            for index in range(len(scores)):
                file.write(
                    f"{name},"
                    f"{index + 1},"
                    f"{scores[index]:.8f},"
                    f"{precision[index]:.8f},"
                    f"{recall[index]:.8f}\n"
                )

    print(f"Saved CSV:  {OUTPUT_CSV}")

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for name, result in all_results.items():
        print(
            f"{name:6s} "
            f"AP: {result['ap'] * 100:6.2f}% "
            f"| Recall @ 0.40: "
            f"{result['recall_04'] * 100:6.2f}%"
        )


if __name__ == "__main__":
    main()
