import enum
from typing import Callable, List, Union

import torch
from toolbox.imageOperators import BlurConvolution


class BlurApplication(enum.IntEnum):
    COMPOSE = 0
    LtSL = 0
    SUM = 1
    LitSLi = 1
    SL = 2
    GRAY = 3


def identity(x: torch.Tensor) -> torch.Tensor:
    return x


def tanh_saturation(x: torch.Tensor) -> torch.Tensor:
    return torch.tanh(2 * x - 1) / 2 + 0.5


def tanh_saturation_hard(x: torch.Tensor) -> torch.Tensor:
    return torch.tanh(0.6 * (2 * x - 1)) / 2 + 0.5


def tanh_saturation_normalised(x: torch.Tensor) -> torch.Tensor:
    return torch.tanh(x)


def tanh_saturation_normalised_hard(x: torch.Tensor) -> torch.Tensor:
    return torch.tanh(0.6 * x)


class DegradationFunction:
    def __init__(
        self,
        Li: Union[BlurConvolution, List[BlurConvolution], None],
        S: Callable[[torch.Tensor], torch.Tensor],
        alphai: Union[float, List[float]],
        type_: BlurApplication,
    ):
        if Li is None or len(Li) == 0:
            print("No blurs provided. Using only saturation")
            self.blurs = []
        else:
            if not isinstance(Li, list):
                self.blurs = [Li]
            else:
                self.blurs = Li
        print(f"Provided {len(self.blurs)} blurs")
        self.saturation_fn = S
        print(f"Using saturation function {S.__name__}")

        self.type_ = type_
        if self.type_ == BlurApplication.LtSL:
            print("Using composition of blurs. Ignoring alpha")
            self.alphai = None
        elif self.type_ == BlurApplication.LitSLi:
            print("Using sum of blurs.")
            if isinstance(alphai, float):
                print("Using same alpha for all blurs")
                self.alphai = [alphai for _ in range(len(self.blurs))]
            else:
                print("Using different alpha for each blur")
                self.alphai = alphai
                assert len(self.alphai) == len(self.blurs)
        elif self.type_ == BlurApplication.SL:
            print("Using sum of blurs. Ignoring alpha")
            self.alphai = None

    def __call__(self, im: torch.Tensor) -> torch.Tensor:
        S = self.saturation_fn
        Lis = self.blurs
        ais = self.alphai

        if len(self.blurs) == 0:
            return S(im)

        if self.type_ == BlurApplication.LtSL:  # Composition of blurs
            y = im.clone()

            for Li in Lis:
                y = Li @ y
            y = S(y)
            for Li in reversed(Lis):
                y = Li.T @ y
            return y

        elif self.type_ == BlurApplication.LitSLi:  # Sum of blurs (~symmetric)
            y = torch.zeros_like(im)

            for Li, a in zip(Lis, ais):
                sl = S(Li @ im)
                y += a * (Li.T @ sl) + (1 - a) * sl
            return y / len(Lis)

        elif self.type_ == BlurApplication.SL:  # Sum of blurs (non-symmetric)
            y = torch.zeros_like(im)

            for Li in Lis:
                y += S(Li @ im)
            return y / len(Lis)

        else:
            raise ValueError(f"Unknown type {self.type_}")


class ColorToGrayDegradation(DegradationFunction):
    def __init__(self):
        super().__init__(
            Li=None,
            S=identity,
            alphai=None,
            type_=BlurApplication.GRAY,
        )

    def __call__(self, im: torch.Tensor) -> torch.Tensor:
        return im.mean(dim=0, keepdim=True).repeat(3, 1, 1)
