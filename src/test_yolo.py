import sys
from pathlib import Path

import torch
from PIL import Image

from darknet import Darknet
from utils import do_detect, plot_boxes


# --------------------------------------------------
# Paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

CFG_PATH = ROOT / "cfg" / "yolo.cfg"
WEIGHTS_PATH = ROOT / "weights" / "yolo.weights"

IMAGE_PATH = ROOT / "data" / "test_images" / "crop001001.png"
OUTPUT_PATH = ROOT / "results" / "images" / "test_detection.png"


# --------------------------------------------------
# Settings
# --------------------------------------------------

CONF_THRESHOLD = 0.4
NMS_THRESHOLD = 0.4

USE_CUDA = torch.cuda.is_available()


# COCO class names
CLASS_NAMES = [
    "person", "bicycle", "car", "motorbike", "aeroplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair",
    "sofa", "pottedplant", "bed", "diningtable", "toilet",
    "tvmonitor", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]


def main():

    print("======================================")
    print("YOLOv2 TEST")
    print("======================================")

    print(f"Device: {'CUDA' if USE_CUDA else 'CPU'}")
    print(f"Config: {CFG_PATH}")
    print(f"Weights: {WEIGHTS_PATH}")
    print(f"Image: {IMAGE_PATH}")
    print()

    # --------------------------------------------------
    # Check files
    # --------------------------------------------------

    if not CFG_PATH.exists():
        print(f"ERROR: Could not find {CFG_PATH}")
        sys.exit(1)

    if not WEIGHTS_PATH.exists():
        print(f"ERROR: Could not find {WEIGHTS_PATH}")
        sys.exit(1)

    if not IMAGE_PATH.exists():
        print(f"ERROR: Could not find {IMAGE_PATH}")
        sys.exit(1)

    # --------------------------------------------------
    # Load YOLO
    # --------------------------------------------------

    print("Loading YOLOv2...")

    model = Darknet(str(CFG_PATH))
    model.load_weights(str(WEIGHTS_PATH))
    model.eval()

    if USE_CUDA:
        model.cuda()

    print("YOLOv2 loaded.")
    print()

    # --------------------------------------------------
    # Load image
    # --------------------------------------------------

    image = Image.open(IMAGE_PATH).convert("RGB")

    print(f"Original image size: {image.size}")

    # Resize image to YOLOv2 network input size
    image = image.resize((model.width, model.height))

    print(f"YOLO input size: {image.size}")


    # --------------------------------------------------
    # Detection
    # --------------------------------------------------

    print("Running detection...")

    boxes = do_detect(
        model,
        image,
        CONF_THRESHOLD,
        NMS_THRESHOLD,
        USE_CUDA
    )
    
    print()
    print(f"Total detections: {len(boxes)}")
    print()

    # --------------------------------------------------
    # Print detections
    # --------------------------------------------------

    person_count = 0

    for i, box in enumerate(boxes):

        # Typical pytorch-yolo2 box:
        #
        # box[0] = center x
        # box[1] = center y
        # box[2] = width
        # box[3] = height
        # box[4] = objectness
        # box[5] = class confidence
        # box[6] = class id

        class_id = int(box[6])

        class_name = CLASS_NAMES[class_id]

        objectness = float(box[4])
        class_conf = float(box[5])

        final_conf = objectness * class_conf

        print(f"Detection {i + 1}")
        print(f"  Class:       {class_name}")
        print(f"  Objectness:  {objectness:.4f}")
        print(f"  Class conf:  {class_conf:.4f}")
        print(f"  Final conf:  {final_conf:.4f}")

        if class_name == "person":
            person_count += 1
            print("  >>> PERSON DETECTED")

        print()

    print("--------------------------------------")
    print(f"Persons detected: {person_count}")
    print("--------------------------------------")

    # --------------------------------------------------
    # Save result
    # --------------------------------------------------

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    plot_boxes(
        image,
        boxes,
        str(OUTPUT_PATH),
        CLASS_NAMES
    )

    print()
    print(f"Saved detection image to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
