from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as TF

from darknet import Darknet
from yolo_loss import YOLOv2PersonLoss


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


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def main():

    print("Loading YOLOv2...")

    model = Darknet(
        str(CFG_PATH)
    )

    model.load_weights(
        str(WEIGHTS_PATH)
    )

    model = model.to(DEVICE)

    model.eval()

    # -----------------------------------------
    # Freeze YOLO weights
    # -----------------------------------------

    for parameter in model.parameters():

        parameter.requires_grad = False

    print("YOLO loaded.")

    # -----------------------------------------
    # Image
    # -----------------------------------------

    image = Image.open(
        IMAGE_PATH
    ).convert("RGB")

    image = image.resize(
        (416, 416)
    )

    image_tensor = TF.to_tensor(
        image
    )

    image_tensor = (
        image_tensor
        .unsqueeze(0)
        .to(DEVICE)
    )

    # We are only testing gradient flow
    image_tensor.requires_grad_(True)

    # -----------------------------------------
    # YOLO
    # -----------------------------------------

    output = model(
        image_tensor
    )

    print(
        "YOLO output:",
        output.shape
    )

    # -----------------------------------------
    # Loss
    # -----------------------------------------

    loss_function = YOLOv2PersonLoss(
        attack_type="obj"
    )

    loss = loss_function(
        output
    )

    print(
        f"Attack score: "
        f"{loss.item():.6f}"
    )

    # -----------------------------------------
    # Backpropagation
    # -----------------------------------------

    loss.backward()

    print()

    if image_tensor.grad is None:

        print("ERROR:")
        print("No gradient reached the image.")

    else:

        print("SUCCESS:")
        print("Gradient reached the image.")

        print(
            "Gradient shape:",
            image_tensor.grad.shape
        )

        print(
            "Mean gradient:",
            image_tensor.grad.abs().mean().item()
        )

        print(
            "Maximum gradient:",
            image_tensor.grad.abs().max().item()
        )


if __name__ == "__main__":
    main()
