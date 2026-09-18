import torch
import torch.nn.functional as F


def create_random_patch(patch_size=100):
    """
    Create a random RGB adversarial patch.

    Output:
        Tensor shape: [3, patch_size, patch_size]

    Pixel values:
        0.0 - 1.0
    """

    patch = torch.rand(
        3,
        patch_size,
        patch_size
    )

    return patch


def resize_patch(patch, target_size):
    """
    Resize patch to target_size x target_size.
    """

    patch = patch.unsqueeze(0)

    patch = F.interpolate(
        patch,
        size=(target_size, target_size),
        mode="bilinear",
        align_corners=False
    )

    return patch.squeeze(0)


def place_patch_on_image(
    image_tensor,
    patch,
    center_x,
    center_y
):
    """
    Place a patch on an image tensor.

    image_tensor:
        [3, H, W]

    patch:
        [3, patch_H, patch_W]

    center_x, center_y:
        pixel coordinates
    """

    image = image_tensor.clone()

    _, image_h, image_w = image.shape
    _, patch_h, patch_w = patch.shape

    x1 = int(center_x - patch_w / 2)
    y1 = int(center_y - patch_h / 2)

    x2 = x1 + patch_w
    y2 = y1 + patch_h

    # Keep patch inside image
    x1 = max(0, x1)
    y1 = max(0, y1)

    x2 = min(image_w, x2)
    y2 = min(image_h, y2)

    patch_width = x2 - x1
    patch_height = y2 - y1

    image[
        :,
        y1:y2,
        x1:x2
    ] = patch[
        :,
        0:patch_height,
        0:patch_width
    ]

    return image
