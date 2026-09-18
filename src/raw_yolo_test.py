from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as TF

from darknet import Darknet


ROOT = Path(__file__).resolve().parent.parent

CFG_PATH = ROOT / "cfg" / "yolo.cfg"
WEIGHTS_PATH = ROOT / "weights" / "yolo.weights"
IMAGE_PATH = ROOT / "data" / "test_images" / "crop001001.png"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def main():

    print("======================================")
    print("RAW YOLO OUTPUT TEST")
    print("======================================")
    print(f"Device: {DEVICE}")
    print()

    # -----------------------------------------
    # Load model
    # -----------------------------------------

    print("Loading YOLOv2...")

    model = Darknet(str(CFG_PATH))
    model.load_weights(str(WEIGHTS_PATH))

    model = model.to(DEVICE)
    model.eval()

    # Freeze YOLO
    for parameter in model.parameters():
        parameter.requires_grad = False

    print("YOLO loaded.")
    print()

    # -----------------------------------------
    # Load image
    # -----------------------------------------

    image = Image.open(
        IMAGE_PATH
    ).convert("RGB")

    print(f"Original image size: {image.size}")

    image = image.resize(
        (416, 416)
    )

    image_tensor = TF.to_tensor(image)

    # [3, 416, 416]
    print(
        f"Tensor before batch dimension: "
        f"{image_tensor.shape}"
    )

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    # [1, 3, 416, 416]
    image_tensor = image_tensor.to(DEVICE)

    print(
        f"Tensor sent into YOLO: "
        f"{image_tensor.shape}"
    )

    # -----------------------------------------
    # Raw forward pass
    # -----------------------------------------

    output = model(image_tensor)

    print()
    print("==============================")
    print("RAW OUTPUT")
    print("==============================")

    print(f"Output type: {type(output)}")

    if isinstance(output, torch.Tensor):

        print(f"Output shape: {output.shape}")
        print(f"Requires grad: {output.requires_grad}")
        print(f"Grad function: {output.grad_fn}")

        print()
        print("Minimum:", output.min().item())
        print("Maximum:", output.max().item())
        print("Mean:", output.mean().item())

    else:
        print("YOLO returned something other than")
        print("a single PyTorch tensor.")
        print()
        print(output)


if __name__ == "__main__":
    main()
