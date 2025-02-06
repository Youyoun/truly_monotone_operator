from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
import torch
from toolbox.jacobian import get_lambda_min_or_max_poweriter, power_method
from toolbox.metrics import PSNR, mean_absolute_error
from toolbox.models import RRDBNet, RUnet, get_model
from toolbox.models.SingleConv import SingleFilterConvolutionProjected
from torchvision.utils import save_image

from .operators import FModel, LSUnetModel, Regularization
from .solver_helper_functions import InverseResult, compute_inverse

POWER_ITER_N = 500


def get_min_max_evs(
    g: Callable, x: torch.Tensor, n_power_iter=500
) -> Tuple[float, float]:
    """
    Get the minimum and maximum eigenvalues of the Jacobian of the generator
    at the point x.
    :param g: The callable function
    :param x: The point
    """
    max_ = get_lambda_min_or_max_poweriter(
        g, x, alpha=5, is_eval=False, biggest=True, n_iter=n_power_iter
    ).detach()
    min_ = get_lambda_min_or_max_poweriter(
        g, x, alpha=max_.item(), is_eval=False, biggest=False, n_iter=n_power_iter
    ).detach()
    return min_.item(), max_.item()


class LoadModel:
    def __init__(self) -> None:
        pass

    @staticmethod
    def get_generator(
        depth: int,
        feature_maps: int,
        in_c: int = 1,
        out_c: int = 1,
        last_activ: str = "Tanh",
        mid_activ: str = "LeakyReLU",
        model_type: str = "cnn",
    ):
        if model_type == "cnn":
            generator = get_model(
                in_c, out_c, feature_maps, depth, last_activ, mid_activ
            )
        elif model_type == "unet":
            runet_depth = depth
            filters_ = [feature_maps // (2**i) for i in reversed(range(runet_depth))]
            print(f"Using feature maps for unet: {filters_}")
            generator = RUnet(
                in_c, filters_, mid_activations=mid_activ, last_activation=last_activ
            )
        elif model_type == "rrdb":
            generator = RRDBNet(in_c, out_c, feature_maps, depth)
        elif model_type == "linear":
            kernel_size = 9
            generator = SingleFilterConvolutionProjected(kernel_size)
        else:
            raise ValueError(f"Model type {model_type} not recognized")
        return generator

    @staticmethod
    def from_args(
        depth: int,
        feature_maps: int,
        weights_file_path: str,
        in_c: int = 1,
        out_c: int = 1,
        last_activ: str = "Tanh",
        mid_activ: str = "LeakyReLU",
        model_type: str = "cnn",
    ) -> torch.nn.Module:
        """
        Load a model from a file, with the provided parameters.
        :param depth: The depth of the model
        :param feature_maps: The feature maps
        :param file_path: The file path
        :param in_c: The number of input channels
        :param out_c: The number of output channels
        :param last_activ: The last activation function
        :param mid_activ: The middle activation function
        :return: The loaded model
        """
        generator = LoadModel.get_generator(
            depth, feature_maps, in_c, out_c, last_activ, mid_activ, model_type
        )
        pkg = torch.load(weights_file_path)
        if "state_dict" in pkg:
            generator.load_state_dict(pkg["state_dict"])
        else:
            generator.load_state_dict(torch.load(weights_file_path))
        generator.cpu()
        return generator

    @staticmethod
    def from_file(ckpt_file) -> torch.nn.Module:
        """
        Load a model from a pkg.
        :param pkg: The pkg
        :return: The loaded model
        """
        pkg = torch.load(ckpt_file)
        print(pkg["opt"])
        model_type = pkg["opt"]["model_type"]
        depth = pkg["opt"]["depth"]
        feature_maps = pkg["opt"]["conv_features"]
        last_activ = pkg["opt"]["last_activation"].value
        mid_activ = pkg["opt"]["mid_activation"].value
        in_c = pkg["opt"]["input_channels"]
        out_c = pkg["opt"]["output_channels"]
        generator = LoadModel.get_generator(
            depth, feature_maps, in_c, out_c, last_activ, mid_activ, model_type
        )
        print("Loading model with the following parameters:")
        print(
            f"Model type: {model_type}, Depth: {depth}, Feature maps: {feature_maps}, Channels: {in_c} -> {out_c}, Last activation: {last_activ}, Middle activation: {mid_activ}"
        )
        if "state_dict" in pkg:
            generator.load_state_dict(pkg["state_dict"])
        else:
            generator.load_state_dict(torch.load(ckpt_file))
        generator.cpu()

        if "linear_model" in pkg:
            print("Found linear model with the unet")
            linear_model = LoadModel.get_generator(1, 1, model_type="linear")
            linear_model.load_state_dict(pkg["linear_model"])
            linear_model.blur_para.requires_grad = False
            linear_model.cpu()
            model = LSUnetModel(generator, linear_model)
            return model
        else:
            return generator


def power_iteration(A: np.array, num_iterations: int) -> float:
    # Ideally choose a random vector
    # To decrease the chance that our vector
    # Is orthogonal to the eigenvector
    b_k = np.random.rand(A.shape[1]) * 10

    for _ in range(num_iterations):
        # calculate the matrix-by-vector product Ab
        b_k1 = A @ b_k

        # calculate the norm
        b_k1_norm = np.linalg.norm(b_k1)

        # re normalize the vector
        b_k = b_k1 / b_k1_norm

    return b_k.T @ (A @ b_k)


def test_conjugate_property(A: np.array, x: np.array, y: np.array) -> float:
    """
    Test the conjugate property of the matrix A i.e. <Ax, y> = <x, A^T y>
    :param A: The matrix
    :param x: The first vector
    :param y: The second vector
    :return: The difference between the two inner products
    """
    return (A @ x).T @ y - x.T @ (A.T @ y)


def denorm(x: torch.Tensor) -> torch.Tensor:
    """
    Denormalize the image i.e. x \in [-1, 1] -> x \in [0, 1]
    The formula is simply x = (x + 1) / 2
    :param x: The image
    :return: The denormalized image
    """
    return x / 2 + 0.5


def normalize(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize the image i.e. x \in [0, 1] -> x \in [-1, 1]
    The formula is simply x = 2 * x - 1
    :param x: The image
    :return: The normalized image
    """
    return 2 * x - 1


def compute_all_results(
    test_ds: torch.utils.data.Dataset,
    non_monotone_model: torch.nn.Module,
    monotone_model: torch.nn.Module,
    linear_model: Callable[[torch.Tensor], torch.Tensor],
    idx: int,
    device: torch.device,
    save_dir: str = "results",
    save_suffix: str = "",
    rerun: bool = False,
    use_tv: bool = False,
    lambda_: float = 0.01,
    n_iter: int = 1000,
    use_least_squares: bool = False,
) -> Dict[str, InverseResult]:
    """
    We compute all the results related to the inversion of a monotone neural network.
    I counted 8 results in total per image
    - Real image
    - Blurred image
    - Not monotone prediction
    - Monotone prediction
    - Inverse of non-monotone model on target of non-monotone model
    - Inverse of non-monotone model on real target
    - Inverse of monotone model on target of monotone model
    - Inverse of monotone model on real target
    Notation:
    - $x$ is the real image
    - $y = F(x)$ is the blurred image
    - $y_{A_{notmon}} = A_{notmon}(x)$ is the prediction of the non-monotone model
    - $y_{A_{mon}} = A_{mon}(x)$ is the prediction of the monotone model
    - $x_{A_{notmon}} = A_{notmon}^{-1}(y_{A_{notmon}})$ is the inverse of the non-monotone model on the target of the non-monotone model
    - $x_{A_{mon}} = A_{mon}^{-1}(y_{A_{mon}})$ is the inverse of the monotone model on the target of the monotone model
    - $x_{A_{notmon}}^* = A_{notmon}^{-1}(F(x))$ is the inverse of the non-monotone model on the real target
    - $x_{A_{mon}}^* = A_{mon}^{-1}(F(x))$ is the inverse of the monotone model on the real target
    """
    save_dir = Path(save_dir) / save_suffix / str(idx)
    save_dir.mkdir(parents=True, exist_ok=True)

    x, y = test_ds[idx]
    x, y = x.unsqueeze(0), y.unsqueeze(0)

    all_results = {}

    for model, name in zip(
        [non_monotone_model, monotone_model, linear_model],
        ["notmonotone", "monotone", "linear"],
    ):
        print(f"Computing Results for {name} model")
        results_dir = save_dir / name
        if results_dir.exists() and not rerun:
            print("Loading results from file")
            inverse_results = InverseResult.load(results_dir)
        else:
            results_dir.mkdir(parents=True, exist_ok=True)
            if name != "linear":
                model.to(device)
            model_reg = Regularization(model, add_tv=use_tv, lambda_tv=lambda_)
            if name == "linear" and use_least_squares:
                print("Using least squares")
                # assert len(model_reg.model.degrad_fn.blurs) == 1
                y_least_squares = model.degrad_fn_T(y)
                inverse_results = compute_inverse(
                    model_reg, y_least_squares, x, n_iter, device=device
                )
            else:
                inverse_results = compute_inverse(
                    model_reg, y, x, n_iter, device=device
                )
            if name != "linear":
                model.to("cpu")
            inverse_results.save(results_dir)
        all_results[name] = inverse_results
    return all_results


title_types = {
    "x": r"$x$",
    "y": r"$y = F(x)$ - PSNR: {:.2f}",
    "y_A": r"$y_{} = {}(x)$"
    "\n"
    r"MAE($y, y_{}$): {:.3f}"
    "\n"
    r"$\lambda_{{min}}$={:.3f} - $\lambda_{{max}}$={:.3f}",
    "x_A": r"$x_{} = {}^{{-1}}(y_{}$)"
    "\n"
    r"MAE($x, x_{}$): {:.3f}"
    "\n"
    r"PSNR: {:.2f}",
    "x_A_star": r"$x_{}^* = {}^{{-1}}(y)$"
    "\n"
    r"MAE($x, x_{}^*$): {:.3f}"
    "\n"
    r"PSNR: {:.2f}",
}


def get_title(title_type, **kwargs):
    if title_type == "x":
        return title_types[title_type]
    elif title_type == "y":
        if "psnr" in kwargs:
            return title_types[title_type].format(kwargs["psnr"])
        elif "y" in kwargs and "x" in kwargs:
            return title_types[title_type].format(PSNR(kwargs["x"], kwargs["y"]))
        else:
            raise ValueError("Missing arguments: y, x")
    elif title_type == "y_A":
        if "y" in kwargs and "y_A" in kwargs and "model" in kwargs:
            model_name = kwargs["model"].__class__.__name__
            if "model_name" in kwargs:
                model_name = kwargs["model_name"]
            if isinstance(kwargs["model"].model, FModel):
                _, max_, _ = power_method(
                    kwargs["y"].unsqueeze(0),
                    lambda u: kwargs["model"].grad(u),
                    POWER_ITER_N,
                )
                max_ = max_.item()
                _, min_, _ = power_method(
                    kwargs["y"].unsqueeze(0),
                    lambda u: max_ * u - kwargs["model"].grad(u),
                    POWER_ITER_N,
                )
                min_ = max_ - min_.item()
            else:
                min_, max_ = get_min_max_evs(
                    kwargs["model"].model, kwargs["y"].unsqueeze(0), POWER_ITER_N
                )
            return title_types[title_type].format(
                model_name,
                model_name,
                model_name,
                mean_absolute_error(kwargs["y"], kwargs["y_A"]),
                min_,
                max_,
            )
        else:
            raise ValueError("Missing arguments: y, x, model")
    elif title_type == "x_A":
        if (
            "x" in kwargs
            and "x_A" in kwargs
            and ("model" in kwargs or "model_name" in kwargs)
        ):
            if "model_name" in kwargs:
                model_name = kwargs["model_name"]
            else:
                model_name = kwargs["model"].__class__.__name__
            return title_types[title_type].format(
                model_name,
                model_name,
                model_name,
                model_name,
                mean_absolute_error(kwargs["x"], kwargs["x_A"]),
                PSNR(kwargs["x"], kwargs["x_A"]),
            )
        else:
            raise ValueError("Missing arguments: x, x_A, model")
    elif title_type == "x_A_star":
        if (
            "x" in kwargs
            and "x_A_star" in kwargs
            and ("model" in kwargs or "model_name" in kwargs)
        ):
            if "model_name" in kwargs:
                model_name = kwargs["model_name"]
            else:
                model_name = kwargs["model"].__class__.__name__
            return title_types[title_type].format(
                model_name,
                model_name,
                model_name,
                mean_absolute_error(kwargs["x"], kwargs["x_A_star"]),
                PSNR(kwargs["x"], kwargs["x_A_star"]),
            )
        else:
            raise ValueError("Missing arguments: x, x_A_star, model")
    else:
        raise ValueError(
            f"Unknown title type {title_type}. Available: {title_types.keys()}"
        )


result_to_title = {
    "x": r"$x$",
    "y": r"$y = F(x)$ - PSNR: {:.2f}",
    "x_A_notmon_star": r"$x_B^* = B^{{-1}}(y)$"
    "\n"
    r"MAE($x, x_{{B}}^*$): {:.3f}"
    "\n"
    r"PSNR: {:.2f}",
    "x_A_notmon": r"$x_B = B^{{-1}}(y_B)$"
    "\n"
    r"MAE($x, x_B$): {:.3f}"
    "\n"
    r"PSNR: {:.2f}",
    "y_A_notmon": r"$y_B = B(x)$"
    "\n"
    r"MAE($y, y_B$): {:.3f}"
    "\n"
    r"$\lambda_{{min}}$={:.3f} - $\lambda_{{max}}$={:.3f}",
    "x_A_mon_star": r"$x_A^* = A^{{-1}}(y)$"
    "\n"
    r"MAE($x, x_A^*$): {:.3f}"
    "\n"
    r"PSNR: {:.2f}",
    "x_A_mon": r"$x_A = A^{{-1}}(y_A)$"
    "\n"
    r"MAE($x, x_A$): {:.3f}"
    "\n"
    r"PSNR: {:.2f}",
    "y_A_mon": r"$y_A = A(x)$"
    "\n"
    r"MAE($y, y_A$): {:.3f}"
    "\n"
    r"$\lambda_{{min}}$={:.3f} - $\lambda_{{max}}$={:.3f}",
    "x_A_lin_star": r"$x_{{F_{{lin}}}}^* = F_{{lin}}^{{-1}}(y)$"
    "\n"
    r"MAE($x, x_{{F_{{lin}}}}^*$): {:.3f}"
    "\n"
    r"PSNR: {:.2f}",
    "x_A_lin": r"$x_{{F_{{lin}}}} = F_{{lin}}^{{-1}}(y_{{F_{{lin}}}})$"
    "\n"
    r"MAE($x, x_{{F_{{lin}}}}$): {:.3f}"
    "\n"
    r"PSNR: {:.2f}",
    "y_A_lin": r"$y_{{F_{{lin}}}} = F_{{lin}}(x)$"
    "\n"
    r"MAE($y, y_{{F_{{lin}}}}$): {:.3f}"
    "\n"
    r"$\lambda_{{min}}$={:.3f} - $\lambda_{{max}}$={:.3f}",
}


def get_all_results_for_plot(
    test_ds: torch.utils.data.Dataset,
    non_monotone_model: torch.nn.Module,
    monotone_model: torch.nn.Module,
    linear_model: Callable[[torch.Tensor], torch.Tensor],
    idx: int,
    device: str,
    save_dir: str = "results",
    save_suffix: str = "",
    short_return: bool = False,
    rerun=False,
    tv_weight: float = 0.0,
    n_iter: int = 2000,
    use_least_squares: bool = False,
) -> Dict[str, Tuple[torch.Tensor, str]]:
    """
    Get all results for a given index using the given models.
    Wrapper around `compute_all_results` and `compute_inverse`.
    """
    res_dict = compute_all_results(
        test_ds,
        non_monotone_model,
        monotone_model,
        linear_model,
        idx,
        device,
        save_dir=save_dir,
        save_suffix=save_suffix,
        rerun=rerun,
        use_tv=tv_weight > 0.0,
        lambda_=tv_weight,
        n_iter=n_iter,
        use_least_squares=use_least_squares,
    )

    if short_return:
        n_power_iter = 1
    else:
        n_power_iter = POWER_ITER_N

    results = {}
    x, y = test_ds[idx]
    x = x.squeeze()
    y = y.squeeze()

    results["x"] = (x, result_to_title["x"])
    blur_psnr = PSNR(x, y)
    results["y"] = (y, result_to_title["y"].format(blur_psnr))

    for model, long_name, short_name in zip(
        [non_monotone_model, monotone_model, linear_model],
        ["notmonotone", "monotone", "linear"],
        ["notmon", "mon", "lin"],
    ):
        if short_name == "lin":  # Need to do this for linear model
            # Linearized
            _, l_max, _ = power_method(
                x.unsqueeze(0), lambda u: model.grad(u), n_power_iter
            )
            l_max = l_max.item()
            _, l_min, _ = power_method(
                x.unsqueeze(0), lambda u: l_max * u - model.grad(u), n_power_iter
            )
            l_min = l_max - l_min.item()

        else:
            if len(x.shape) == 2:
                l_min, l_max = get_min_max_evs(
                    model, x.unsqueeze(0).unsqueeze(0), n_power_iter
                )
            else:
                l_min, l_max = get_min_max_evs(model, x.unsqueeze(0), n_power_iter)

        results[f"y_A_{short_name}"] = (
            res_dict[long_name].y_star,
            result_to_title[f"y_A_{short_name}"].format(
                mean_absolute_error(res_dict[long_name].y_star, y),
                l_min,
                l_max,
            ),
        )
        results[f"x_A_{short_name}"] = (
            res_dict[long_name].x_star,
            result_to_title[f"x_A_{short_name}"].format(
                mean_absolute_error(res_dict[long_name].x_star, x),
                PSNR(x, res_dict[long_name].x_star),
            ),
        )
        results[f"metrics_{short_name}"] = (
            res_dict[long_name].metrics_star,
            r"$\frac{||x_{k+1} - x_k||_2}{||y||_2}$",
        )

        results[f"x_A_{short_name}_star"] = (
            res_dict[long_name].x_hat,
            result_to_title[f"x_A_{short_name}_star"].format(
                mean_absolute_error(res_dict[long_name].x_hat, x),
                PSNR(x, res_dict[long_name].x_hat),
            ),
        )
        results[f"metrics_{short_name}_star"] = (
            res_dict[long_name].metrics_hat,
            r"$\frac{||x_{k+1} - x_k||_2}{||y||_2}$",
        )

    return results


def save_results(results: Dict[str, Tuple[torch.Tensor, str]], save_dir: Path):
    save_dir.mkdir(exist_ok=True, parents=True)
    for title, (im, _) in results.items():
        if torch.is_tensor(im):
            save_image(denorm(im), save_dir / f"{title}.png")
