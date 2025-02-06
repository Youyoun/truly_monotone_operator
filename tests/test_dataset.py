import functools
from typing import Tuple

import torch
import torch.utils.data as tdata
from toolbox.dataset.generic_dataset import GenericDataset
from toolbox.imageOperators import BlurConvolution, Kernels
from torch.utils import data as torch_data
from torchvision import transforms as transforms

from dataset import Saturation, SumLiTSLi, LTSL, TRAIN_DATA, TEST_DATA

normalize = transforms.Normalize((0.5,), (0.5,))


def denorm(arr: torch.Tensor) -> torch.Tensor:
    return (arr + 1) / 2


def saturation(im: torch.Tensor, level: int = 5, use_blur: bool = False) -> torch.Tensor:
    if not use_blur:
        return denorm(torch.tanh(normalize(im) * level))
    else:
        blur = BlurConvolution(3, Kernels.UNIFORM, 0.0)
        return blur.T @ denorm(torch.tanh(normalize(blur @ im) * level))


def double_blur_saturation(im: torch.Tensor) -> torch.Tensor:
    blur_1 = BlurConvolution(9, Kernels.GAUSSIAN, (1.0, 0.5))
    blur_2 = BlurConvolution(9, Kernels.GAUSSIAN, (0.5, 1.0))
    return blur_2.T @ (blur_1.T @ denorm(torch.tanh(normalize(blur_2 @ (blur_1 @ im)))))


def sum_two_blur_saturation(im: torch.Tensor) -> torch.Tensor:
    blur_1 = BlurConvolution(9, Kernels.GAUSSIAN, (2.0, 1.0))
    blur_2 = BlurConvolution(9, Kernels.GAUSSIAN, (1.0, 2.0))
    return 1 / 2 * (blur_1.T @ denorm(torch.tanh(normalize(blur_1 @ im))) + blur_2.T @ denorm(
        torch.tanh(normalize(blur_2 @ im))))


def sum_three_blur_saturation(im: torch.Tensor) -> torch.Tensor:
    blur_1 = BlurConvolution(9, Kernels.GAUSSIAN, (2.0, 1.0))
    blur_2 = BlurConvolution(9, Kernels.GAUSSIAN, (1.0, 2.0))
    blur_3 = BlurConvolution(9, Kernels.GAUSSIAN, (1.0, 1.0))
    return 1 / 3 * (blur_1.T @ denorm(torch.tanh(normalize(blur_1 @ im))) + blur_2.T @ denorm(
        torch.tanh(normalize(blur_2 @ im))) + blur_3.T @ denorm(torch.tanh(normalize(blur_3 @ im))))


def sum_n_blur_saturation(im: torch.Tensor, blur_parameters) -> torch.Tensor:
    blurs = [BlurConvolution(9, Kernels.GAUSSIAN, (b_1, b_2)) for (b_1, b_2) in blur_parameters]
    return 1 / len(blurs) * sum(blur.T @ denorm(torch.tanh(normalize(blur @ im))) for blur in blurs)


def get_dataset(crop_size: int,
                saturation_level: float,
                batch_size: int,
                use_blur: bool = False,
                saturation_function: str = "single",
                random_blur_parameters=None) -> Tuple[GenericDataset,
                                                      tdata.DataLoader,
                                                      GenericDataset,
                                                      tdata.DataLoader]:
    if saturation_function == "single":
        saturation_fn = functools.partial(saturation, level=saturation_level, use_blur=use_blur)
    elif saturation_function == "double":
        saturation_fn = double_blur_saturation
    elif saturation_function == "sum_two":
        saturation_fn = sum_two_blur_saturation
    elif saturation_function == "sum_three":
        saturation_fn = sum_three_blur_saturation
    elif saturation_function == "sum_n_blur":
        saturation_fn = functools.partial(sum_n_blur_saturation, blur_parameters=random_blur_parameters)
    else:
        raise ValueError("Unknown saturation function")
    dataset_transforms = [("centercrop", {"size": crop_size}), ("normalize", {"mean": 0.5, "std": 0.5})]
    ds = GenericDataset(TRAIN_DATA, 0, saturation_fn, augments=dataset_transforms, load_in_memory=True)
    # print("Loading only test set on memory")
    test_ds = GenericDataset(TEST_DATA, 0, saturation_fn, augments=dataset_transforms, load_in_memory=True)
    dl = torch_data.DataLoader(ds, batch_size=batch_size, shuffle=True)
    test_dl = torch_data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return ds, dl, test_ds, test_dl


class TestDegradationFuncs:
    def test_get_dataset_single_no_blur(self):
        _, _, ds, _ = get_dataset(128, 1.0, 1, use_blur=False, saturation_function="single")
        degrad_fn = Saturation()
        for idx in range(len(ds)):
            im = ds.images[idx]
            degraded = ds.noisy_images[idx]
            assert (degrad_fn(im) == degraded).all()

    def test_get_dataset_single_blur(self):
        _, _, ds, _ = get_dataset(128, 1.0, 1, use_blur=True, saturation_function="single")
        blur = BlurConvolution(3, Kernels.UNIFORM, 0.0)
        degrad_fn_1 = SumLiTSLi(blur)
        degrad_fn_2 = LTSL(blur)
        for idx in range(len(ds)):
            im = ds.images[idx]
            degraded = ds.noisy_images[idx]
            assert (degrad_fn_1(im) == degraded).all()
            assert (degrad_fn_2(im) == degraded).all()

    def test_get_dataset_double_blur(self):
        _, _, ds, _ = get_dataset(128, 1.0, 1, use_blur=True, saturation_function="double")
        blur_1 = BlurConvolution(9, Kernels.GAUSSIAN, (1.0, 0.5))
        blur_2 = BlurConvolution(9, Kernels.GAUSSIAN, (0.5, 1.0))
        degrad_fn = LTSL([blur_1, blur_2])
        for idx in range(len(ds)):
            im = ds.images[idx]
            degraded = ds.noisy_images[idx]
            assert (degrad_fn(im) == degraded).all()

    def test_get_dataset_sum_two(self):
        _, _, ds, _ = get_dataset(128, 1.0, 1, use_blur=True, saturation_function="sum_two")
        blur_1 = BlurConvolution(9, Kernels.GAUSSIAN, (2.0, 1.0))
        blur_2 = BlurConvolution(9, Kernels.GAUSSIAN, (1.0, 2.0))
        degrad_fn = SumLiTSLi([blur_1, blur_2])
        for idx in range(len(ds)):
            im = ds.images[idx]
            degraded = ds.noisy_images[idx]
            assert (degrad_fn(im) == degraded).all()

    def test_get_dataset_sum_three(self):
        _, _, ds, _ = get_dataset(128, 1.0, 1, use_blur=True, saturation_function="sum_three")
        blur_1 = BlurConvolution(9, Kernels.GAUSSIAN, (2.0, 1.0))
        blur_2 = BlurConvolution(9, Kernels.GAUSSIAN, (1.0, 2.0))
        blur_3 = BlurConvolution(9, Kernels.GAUSSIAN, (1.0, 1.0))
        degrad_fn = SumLiTSLi([blur_1, blur_2, blur_3])
        for idx in range(len(ds)):
            im = ds.images[idx]
            degraded = ds.noisy_images[idx]
            assert torch.isclose(degrad_fn(im), degraded).all(), (degraded - degrad_fn(im)).max()

    def test_get_dataset_sum_n_blur(self):
        random_blur_parameters = [((torch.rand(1) + 0.5).item(), (torch.rand(1) + 0.5).item()) for _ in range(15)]
        _, _, ds, _ = get_dataset(128, 1.0, 1, use_blur=True, saturation_function="sum_n_blur",
                                  random_blur_parameters=random_blur_parameters)
        blurs = [BlurConvolution(9, Kernels.GAUSSIAN, (b_1, b_2)) for (b_1, b_2) in random_blur_parameters]
        degrad_fn = SumLiTSLi(blurs)
        for idx in range(len(ds)):
            im = ds.images[idx]
            degraded = ds.noisy_images[idx]
            assert torch.isclose(degrad_fn(im), degraded).all(), (degraded - degrad_fn(im)).max()
