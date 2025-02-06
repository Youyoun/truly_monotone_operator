from pathlib import Path
from typing import Union

import numpy as np
import torch
from matplotlib import pyplot as plt
from toolbox.metrics import mean_absolute_error

from restoration.utils import denorm

DPI = 160
VMIN = -1


def create_log_dir(dir_path: Path) -> None:
    dir_path.mkdir(exist_ok=True, parents=True)
    images_dir = dir_path / "images"
    images_dir.mkdir()


def plot_img(
    im_array: Union[torch.Tensor, np.array],
    title: str,
    ax: plt.Axes,
    do_crop: bool = False,
) -> plt.Axes:
    if do_crop:
        ax.imshow(im_array, cmap="gray", vmin=VMIN, vmax=1)
    else:
        ax.imshow(im_array, cmap="gray")
    ax.set_title(title)
    ax.axis("off")
    return ax


def tensor2array(img, is_image_normalized: bool = False):
    img = img.cpu()
    img = img.squeeze().detach().numpy()
    if len(img.shape) == 3:
        img = np.transpose(img, (1, 2, 0))
        if is_image_normalized:
            img = denorm(img)
    return img


def log_example_image(
    input_: torch.Tensor, truth: torch.Tensor, im_file_path: Path
) -> None:
    fig, ax = plt.subplots(1, 2)
    plot_img(tensor2array(input_), "Original", ax[0], False)
    plot_img(tensor2array(truth), "Saturated", ax[1], False)
    plt.tight_layout(pad=0)
    fig.savefig(im_file_path, bbox_inches="tight", dpi=DPI)
    plt.close(fig)


def log_result(
    input_: torch.Tensor,
    truth: torch.Tensor,
    predicted: torch.Tensor,
    sample_idx: int,
    save_path: Path,
    is_image_normalized: bool = False,
):
    fig, ax = plt.subplots(1, 4)
    plot_img(
        tensor2array(input_, is_image_normalized),
        f"Original {sample_idx}",
        ax[0],
        False,
    )
    plot_img(
        tensor2array(truth, is_image_normalized),
        f"Saturated {sample_idx}",
        ax[1],
        False,
    )
    l1 = mean_absolute_error(predicted, truth)
    plot_img(
        tensor2array(predicted, is_image_normalized),
        f"Predicted - L1: {l1:.2f}",
        ax[2],
        False,
    )
    plot_img(
        np.abs(tensor2array(predicted - truth)).mean(axis=2), "Difference", ax[3], False
    )
    plt.tight_layout(pad=0)
    fig.savefig(save_path, bbox_inches="tight", dpi=DPI)
    plt.close(fig)
