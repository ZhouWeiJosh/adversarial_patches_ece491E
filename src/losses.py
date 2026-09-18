import torch


def total_variation(patch):
    """
    Encourage neighboring pixels to
    have similar values.
    """

    horizontal = torch.abs(
        patch[:, :, 1:]
        -
        patch[:, :, :-1]
    )

    vertical = torch.abs(
        patch[:, 1:, :]
        -
        patch[:, :-1, :]
    )

    tv = (
        horizontal.mean()
        +
        vertical.mean()
    )

    return tv
