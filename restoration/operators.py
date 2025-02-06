from typing import Callable, List

import torch
from toolbox.base_classes import Fidelity
from toolbox.base_classes import Regularization as BaseRegularization
from toolbox.imageOperators import BlurConvolution, SmoothTotalVariation

from degradation import (
    BlurApplication,
    DegradationFunction,
    identity,
    tanh_saturation,
    tanh_saturation_hard,
)


class InverseFidelity(Fidelity):
    def f(self, x: torch.Tensor, y: torch.Tensor):
        return (x * y).sum().sqrt()

    def grad(self, x: torch.Tensor, y: torch.Tensor):
        return -y


class Regularization(BaseRegularization):
    def __init__(
        self,
        model: Callable[[torch.Tensor], torch.Tensor],
        add_tv: bool = False,
        lambda_tv: float = 0.1,
    ):
        super().__init__()
        self.model = model
        self.tv_reg = None
        if add_tv:
            self.tv_reg = SmoothTotalVariation()
        self.lambda_tv = lambda_tv

    def f(self, x: torch.Tensor):
        eval_ = torch.zeros(1)
        if self.tv_reg is not None:
            eval_ += self.lambda_tv * self.tv_reg.f(x)
        return eval_

    def grad(self, x: torch.Tensor):
        with torch.no_grad():
            if self.tv_reg is not None:
                grad_ = self.model(x) + self.lambda_tv * self.tv_reg.grad(x)
            else:
                grad_ = self.model(x)
        return grad_


class FModel(BaseRegularization):
    def __init__(
        self,
        blurs: List[BlurConvolution],
        blur_mode: BlurApplication,
        is_linear: bool = False,
        use_least_squares: bool = False,
        use_hard_tanh: bool = False,
    ):
        super().__init__()
        self.blurs = blurs
        if is_linear:
            print("Using linear model")
            self.non_linearity = identity
        else:
            print("Using non linear model")
            if use_hard_tanh:
                print("Using hard tanh")
                self.non_linearity = tanh_saturation_hard
            else:
                self.non_linearity = tanh_saturation

        self.degrad_fn = DegradationFunction(
            self.blurs,
            self.non_linearity,
            0.0 if blur_mode == BlurApplication.SL else 1.0,
            blur_mode,
        )
        self.degrad_fn_T = None
        self.least_squares = use_least_squares and blur_mode == BlurApplication.SL
        if self.least_squares:
            print("Using least squares regularization")
            self.degrad_fn_T = DegradationFunction(
                [Lk.T for Lk in self.blurs],
                identity,
                0.0 if blur_mode == BlurApplication.SL else 1.0,
                blur_mode,
            )

    def f(self, x: torch.Tensor):
        return torch.zeros(1).to(x.device)

    def grad(self, x: torch.Tensor):
        with torch.no_grad():
            grad_ = self.degrad_fn(x)
            if self.degrad_fn_T is not None:
                grad_ = self.degrad_fn_T(grad_)
        return grad_


class LinearModel(FModel):
    def __init__(
        self,
        blurs: List[BlurConvolution],
        blur_mode: BlurApplication,
        use_least_squares: bool = False,
        use_hard_tanh: bool = False,
    ):
        super().__init__(
            blurs,
            blur_mode,
            is_linear=True,
            use_least_squares=use_least_squares,
            use_hard_tanh=use_hard_tanh,
        )


class NonLinearModel(FModel):
    def __init__(
        self,
        blurs: List[BlurConvolution],
        blur_mode: BlurApplication,
        use_least_squares: bool = False,
        use_hard_tanh: bool = False,
    ):
        super().__init__(
            blurs,
            blur_mode,
            is_linear=False,
            use_least_squares=use_least_squares,
            use_hard_tanh=use_hard_tanh,
        )


class LSUnetModel(torch.nn.Module):
    def __init__(self, unet, linear_model):
        super().__init__()
        self.unet = unet
        self.linear_model = linear_model

    def forward(self, x):
        return self.linear_model.forward_t(self.unet(x))
