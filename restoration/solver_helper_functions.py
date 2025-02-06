from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Tuple, Union

import torch
from toolbox.algorithms.proj_op import Identity, Indicator
from toolbox.algorithms.tseng_descent import TsengDescent
from toolbox.base_classes import Regularization as BaseRegularization
from toolbox.metrics import MetricsDictionary
from toolbox.models.SingleConv import SingleFilterConvolutionProjected
from torchvision.utils import save_image

from restoration.operators import FModel, InverseFidelity, LSUnetModel, Regularization


def torch_load_if_exists_or_none(path: Path) -> Union[torch.Tensor, None]:
    if path.exists():
        return torch.load(path)
    else:
        return None


@dataclass
class InverseResult:
    x: Union[torch.Tensor, None]  # The real image
    y: Union[torch.Tensor, None]  # The observed image
    y_star: Union[torch.Tensor, None]  # The degraded image with the operator
    x_hat: Union[torch.Tensor, None]  # The restored image from y
    metrics_hat: Union[MetricsDictionary, None]
    x_star: Union[torch.Tensor, None]  # The restored image from y_star
    metrics_star: Union[MetricsDictionary, None]

    def save(self, path: Path) -> None:
        if self.x is not None:
            torch.save(self.x, path / "_x.pt")
            save_image(self.x, path / "_x.png")
        if self.y is not None:
            torch.save(self.y, path / "_y.pt")
            save_image(self.y, path / "_y.png")
        if self.y_star is not None:
            torch.save(self.y_star, path / "_y_star.pt")
            save_image(self.y_star, path / "_y_star.png")
        if self.x_hat is not None:
            torch.save(self.x_hat, path / "_x_hat.pt")
            save_image(self.x_hat, path / "_x_hat.png")
        if self.x_star is not None:
            torch.save(self.x_star, path / "_x_star.pt")
            save_image(self.x_star, path / "_x_star.png")
        if self.metrics_star is not None:
            self.metrics_star.save(path / "_metrics_star.json")
        if self.metrics_hat is not None:
            self.metrics_hat.save(path / "_metrics_hat.json")

    @classmethod
    def load(cls, path: Path) -> "InverseResult":
        if isinstance(path, str):
            path = Path(path)

        return cls(
            torch_load_if_exists_or_none(path / "_x.pt"),
            torch_load_if_exists_or_none(path / "_y.pt"),
            torch_load_if_exists_or_none(path / "_y_star.pt"),
            torch_load_if_exists_or_none(path / "_x_hat.pt"),
            MetricsDictionary.load(path / "_metrics_hat.json"),
            torch_load_if_exists_or_none(path / "_x_star.pt"),
            MetricsDictionary.load(path / "_metrics_star.json"),
        )


def compute_inverse(
    regularization: BaseRegularization,
    y: torch.Tensor,
    real_x: torch.Tensor,
    n_iters: int = 2000,
    device: str = "cpu",
    compute_inverse_on_model: bool = True,
    use_indicator_fn: bool = True,
) -> InverseResult:
    """
    Compute the inverse of the operator using the Tseng descent algorithm.
    :param regularization: The regularization operator
    :param y: The observed image
    :param real_x: The real image
    :param n_iters: The number of iterations
    :param device: The device to use
    :param compute_inverse_on_model: Whether to compute the inverse on the model or on the degraded image
    :param use_indicator_fn: Whether to use the indicator function or not
    :return: The inverse result
    """

    fid = InverseFidelity()
    solver = TsengDescent(
        fid,
        regularization,
        gamma=1.0,
        lambda_=1.0,
        use_armijo=True,
        max_iter=n_iters,
        device=device,
        indicator_fn=Indicator(0, 1) if use_indicator_fn else Identity(),
        n_step_test_monotony=0,
    )
    x, metrics = solver.solve(y, real_x)
    if compute_inverse_on_model:
        with torch.no_grad():
            y_star = regularization.grad(real_x.to(device)).cpu()
        x_star, metrics_star = solver.solve(y_star, x)
    else:
        y_star = None
        x_star = None
        metrics_star = None
    return InverseResult(
        real_x.squeeze() if real_x is not None else None,
        y.squeeze(),
        y_star.squeeze() if y_star is not None else None,
        x.squeeze(),
        metrics,
        x_star.squeeze() if x_star is not None else None,
        metrics_star,
    )


def restore_image(
    degraded_image: torch.Tensor,
    model: Callable[[torch.Tensor], torch.Tensor],
    device: str,
    lambda_: float = 0.1,
    n_iters: int = 1000,
    use_indicator_fn: bool = False,
    use_least_squares: bool = False,
) -> Tuple[torch.Tensor, Any]:
    """
    Restore an image using the Tseng descent algorithm.
    :param degraded_image: The degraded image
    :param model: The model to use
    :param device: The device to use
    :param lambda_: The lambda parameter
    :param n_iters: The number of iterations
    :param use_indicator_fn: Whether to use the indicator function or not
    """

    img = degraded_image.to(device)
    reg = Regularization(model, add_tv=True, lambda_tv=lambda_)
    if use_least_squares:
        print("Using least squares")
        if isinstance(model, FModel):
            # assert len(model.degrad_fn.blurs) == 1
            if model.degrad_fn_T is None:
                raise ValueError("Need to provide the transpose of the degradation")
            y_least_squares = model.degrad_fn_T(img)
        elif isinstance(model, LSUnetModel):
            y_least_squares = model.linear_model.forward_t(img.unsqueeze(0))
        elif isinstance(model, SingleFilterConvolutionProjected):
            y_least_squares = model.forward_t(img.unsqueeze(0))
        else:
            raise NotImplementedError("Unknown model")
        results = compute_inverse(
            reg,
            y_least_squares,
            None,
            n_iters,
            device=device,
            compute_inverse_on_model=False,
            use_indicator_fn=use_indicator_fn,
        )
    else:
        results = compute_inverse(
            reg, img, None, n_iters, device, False, use_indicator_fn
        )
    return results.x_hat, results.metrics_hat
