from pathlib import Path
import time

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from darknet import Darknet
from utils import do_detect


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

CFG_PATH = ROOT / "cfg" / "yolo.cfg"
WEIGHTS_PATH = ROOT / "weights" / "yolo.weights"
PATCH_PATH = ROOT / "patches" / "universal_patch_obj.png"


# ============================================================
# SETTINGS
# ============================================================

YOLO_SIZE = 416
PATCH_SIZE = 125
TORSO_POSITION = 0.40

CONF_THRESHOLD = 0.40
NMS_THRESHOLD = 0.40

CAMERA_INDEX = 0

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
    return get_value(box[4]) * get_value(box[5])


def get_person_boxes(boxes):
    return [box for box in boxes if int(box[6]) == 0]


def run_detection(model, pil_image):
    return do_detect(
        model,
        pil_image,
        CONF_THRESHOLD,
        NMS_THRESHOLD,
        USE_CUDA,
    )


def box_to_pixels(box, width, height):
    cx = get_value(box[0]) * width
    cy = get_value(box[1]) * height
    bw = get_value(box[2]) * width
    bh = get_value(box[3]) * height

    x1 = int(cx - bw / 2)
    y1 = int(cy - bh / 2)
    x2 = int(cx + bw / 2)
    y2 = int(cy + bh / 2)

    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))

    return x1, y1, x2, y2


def draw_person_boxes(frame, boxes):
    height, width = frame.shape[:2]

    for box in get_person_boxes(boxes):
        confidence = person_confidence(box)

        x1, y1, x2, y2 = box_to_pixels(
            box,
            width,
            height,
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            f"person {confidence:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return frame


def apply_patch_on_person(image_tensor, patch, person_box):
    """
    image_tensor: [3,H,W]
    patch:        [3,P,P]

    For the DIGITAL demo only.
    """

    patched = image_tensor.clone()

    _, image_h, image_w = patched.shape

    person_center_x = get_value(person_box[0]) * image_w
    person_center_y = get_value(person_box[1]) * image_h
    person_height = get_value(person_box[3]) * image_h

    person_top = person_center_y - person_height / 2.0

    patch_center_x = person_center_x
    patch_center_y = (
        person_top
        + TORSO_POSITION * person_height
    )

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

    patched[
        :,
        image_y1:image_y2,
        image_x1:image_x2,
    ] = patch[
        :,
        patch_y1:patch_y2,
        patch_x1:patch_x2,
    ]

    return patched


def apply_patch_to_all_people(image_tensor, patch, clean_boxes):
    patched = image_tensor.clone()

    for person_box in get_person_boxes(clean_boxes):
        patched = apply_patch_on_person(
            patched,
            patch,
            person_box,
        )

    return patched


def add_panel_title(frame, title):
    cv2.rectangle(
        frame,
        (0, 0),
        (frame.shape[1], 35),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        frame,
        title,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return frame


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("YOLOv2 ADVERSARIAL PATCH LIVE DEMO")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print("Press Q to quit.")
    print()

    if not PATCH_PATH.exists():
        print(f"ERROR: patch not found:\n{PATCH_PATH}")
        return

    print("Loading YOLOv2...")

    model = Darknet(str(CFG_PATH))
    model.load_weights(str(WEIGHTS_PATH))
    model.eval()

    if USE_CUDA:
        model.cuda()

    print("YOLO loaded.")

    patch_image = (
        Image.open(PATCH_PATH)
        .convert("RGB")
        .resize((PATCH_SIZE, PATCH_SIZE))
    )

    patch = TF.to_tensor(patch_image)

    camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        print(
            f"ERROR: could not open webcam index "
            f"{CAMERA_INDEX}"
        )
        return

    previous_time = time.time()

    while True:
        ok, frame = camera.read()

        if not ok:
            print("Could not read camera frame.")
            break

        # Mirror for a natural webcam display.
        frame = cv2.flip(frame, 1)

        # YOLO input is 416x416 in your current project.
        resized_bgr = cv2.resize(
            frame,
            (YOLO_SIZE, YOLO_SIZE),
        )

        clean_rgb = cv2.cvtColor(
            resized_bgr,
            cv2.COLOR_BGR2RGB,
        )

        clean_pil = Image.fromarray(clean_rgb)

        # ----------------------------------------------------
        # CLEAN DETECTION
        # ----------------------------------------------------

        clean_boxes = run_detection(
            model,
            clean_pil,
        )

        clean_display = resized_bgr.copy()
        clean_display = draw_person_boxes(
            clean_display,
            clean_boxes,
        )

        # ----------------------------------------------------
        # DIGITAL PATCH
        # ----------------------------------------------------

        clean_tensor = TF.to_tensor(clean_pil)

        patched_tensor = apply_patch_to_all_people(
            clean_tensor,
            patch,
            clean_boxes,
        )

        patched_pil = TF.to_pil_image(
            patched_tensor.clamp(0.0, 1.0)
        )

        patched_rgb = np.array(patched_pil)
        patched_bgr = cv2.cvtColor(
            patched_rgb,
            cv2.COLOR_RGB2BGR,
        )

        patched_boxes = run_detection(
            model,
            patched_pil,
        )

        patched_display = patched_bgr.copy()
        patched_display = draw_person_boxes(
            patched_display,
            patched_boxes,
        )

        # ----------------------------------------------------
        # LABELS + FPS
        # ----------------------------------------------------

        clean_display = add_panel_title(
            clean_display,
            "CLEAN",
        )

        patched_display = add_panel_title(
            patched_display,
            "PATCHED",
        )

        current_time = time.time()
        fps = 1.0 / max(
            current_time - previous_time,
            1e-6,
        )
        previous_time = current_time

        cv2.putText(
            patched_display,
            f"FPS: {fps:.1f}",
            (10, YOLO_SIZE - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # ----------------------------------------------------
        # SIDE-BY-SIDE
        # ----------------------------------------------------

        combined = np.hstack(
            (
                clean_display,
                patched_display,
            )
        )

        cv2.imshow(
            "Adversarial Patch Demo - Q to quit",
            combined,
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
