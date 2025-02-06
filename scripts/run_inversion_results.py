import argparse
import time
from pathlib import Path

from toolbox.models.SingleConv import SingleFilterConvolutionProjected

from dataset import degrad_blur_parameters, get_dataset
from degradation import BlurApplication
from restoration.operators import (
    LinearModel,
    LSUnetModel,
    NonLinearModel,
    Regularization,
)
from restoration.plot_functions import plot_results
from restoration.solver_helper_functions import compute_inverse
from restoration.utils import LoadModel, get_all_results_for_plot, save_results

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_type",
    type=str,
    choices=["linear", "non_linear", "nn"],
    required=True,
)
parser.add_argument("--model_path", type=str)
parser.add_argument("--n_images", type=int, default=1)
parser.add_argument(
    "--blur_kernels", choices=list(degrad_blur_parameters.keys()), required=True
)
parser.add_argument(
    "--blur_mode",
    type=int,
    choices=[b.value for b in BlurApplication],
    help="0: LtSL, 1: LitSLi, 2: SL",
    required=True,
)
parser.add_argument("--use_hard_tanh", action="store_true")
parser.add_argument("--colorized", action="store_true")
parser.add_argument("--n_iters", type=int, default=2000)
parser.add_argument("--device", type=str, default="cuda", required=True)
parser.add_argument("--use_least_squares", action="store_true")
parser.add_argument("--save_folder_name", type=str, default="results/inversion")

if __name__ == "__main__":
    args = parser.parse_args()

    save_dir = Path(args.save_folder_name)
    save_dir = (
        save_dir
        / args.model_type
        / args.blur_kernels
        / str(args.blur_mode)
        / f'{"tanh" if not args.use_hard_tanh else "hard_tanh"}'
        / f'{"ls" if args.use_least_squares else "no_ls"}'
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save args
    with open(save_dir / "args.txt", "w") as f:
        f.write(str(args))

    blurs = degrad_blur_parameters[args.blur_kernels]

    # We will loop through the images of the dataset directly
    # so the arguments are not necessary
    _, _, test_ds, _ = get_dataset(
        "bsd",
        "bsd",
        64,
        1,
        blurs,
        args.n_images,
        BlurApplication.SL,
        return_test_only=True,
        colorized=args.colorized,
        no_crop=False,
        use_hard_saturation=args.use_hard_tanh,
    )

    # Used for generating the data
    non_linear_model = NonLinearModel(
        blurs,
        BlurApplication.SL,
        use_least_squares=False,
        use_hard_tanh=args.use_hard_tanh,
    )

    if args.model_type == "linear":
        print("Computing the results for the linear model")
        model = LinearModel(
            blurs,
            BlurApplication.SL,
            use_least_squares=args.use_least_squares,
            use_hard_tanh=args.use_hard_tanh,
        )

    elif args.model_type == "non_linear":
        print("Computing the results for the non linear model")
        model = NonLinearModel(
            blurs,
            BlurApplication.SL,
            use_least_squares=args.use_least_squares,
            use_hard_tanh=args.use_hard_tanh,
        )

    elif args.model_type == "nn":
        print("Computing the results for the nn model")
        model = LoadModel.from_file(args.model_path)
        model.eval()
        if args.device == "cuda":
            model = model.cuda()
        if isinstance(model, LSUnetModel):
            assert args.use_least_squares, "LSUnetModel requires least squares"
        if isinstance(model, SingleFilterConvolutionProjected):
            assert (
                args.use_least_squares
            ), "SingleFilterConvolutionProjected requires least squares"
            model.blur_para.requires_grad = False
            model.use_symmetric_form = True

    else:
        raise ValueError("Unknown model type")

    regularizer = Regularization(model, add_tv=False)

    for idx in range(len(test_ds)):
        print(f"Processing {idx}")
        start = time.time()
        x = test_ds.images[idx][..., 1:, 1:]
        y = non_linear_model(x)
        x, y = x.unsqueeze(0), y.unsqueeze(0)
        if args.use_least_squares:
            if args.model_type in {"linear", "non_linear"}:
                if model.degrad_fn_T is None:
                    raise ValueError("Need to provide the transpose of the degradation")
                y = model.degrad_fn_T(y)
            if isinstance(model, LSUnetModel):
                y = model.linear_model.forward_t(y)
            if isinstance(model, SingleFilterConvolutionProjected):
                y = model.forward_t(y)
        results = compute_inverse(regularizer, y, x, args.n_iters, args.device)
        print(f"Processed in {time.time() - start}")

        # Save the images
        im_save_dir = save_dir / f"im_{idx:02d}"
        im_save_dir.mkdir(parents=True, exist_ok=True)
        results.save(im_save_dir)
