import argparse
import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import optuna
import torch
from toolbox.metrics import PSNR, SSIM
from toolbox.models.SingleConv import SingleFilterConvolutionProjected

from dataset import degrad_blur_parameters, get_dataset
from degradation import BlurApplication
from restoration.operators import LinearModel, NonLinearModel
from restoration.plot_functions import plot_im
from restoration.solver_helper_functions import restore_image
from restoration.utils import LoadModel, save_image

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default=None, type=str)
parser.add_argument("--dataset", type=str, default="bsd")
parser.add_argument("--image_size", type=int, default=128)
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
parser.add_argument("--colorized", action="store_true")
parser.add_argument("--save_folder_name", type=str, required=True)
parser.add_argument("--n_iters", type=int, default=2000)
parser.add_argument("--device", type=str, default="cuda", required=True)
parser.add_argument("--noise_path", type=str, default=None)
parser.add_argument(
    "--model_type",
    type=str,
    choices=["cnn", "unet", "linear", "nonlinear"],
    required=True,
)
parser.add_argument("--use_least_squares", action="store_true")
parser.add_argument(
    "--noise_levels", nargs="+", type=float, default=[3 / 255, 10 / 255]
)
parser.add_argument(
    "--lambda_min", type=float, default=0.005, help="Minimum lambda value"
)
parser.add_argument(
    "--lambda_max", type=float, default=0.1, help="Maximum lambda value"
)
parser.add_argument(
    "--lambda_tries", type=int, default=10, help="Number of lambda values to try"
)
parser.add_argument("--use_optuna", action="store_true")
parser.add_argument("--parallel", action="store_true")
parser.add_argument("--n_jobs", type=int, default=4)
parser.add_argument("--use_hard_saturation", action="store_true")


def save_im(im, title, name, directory):
    fig, ax = plt.subplots(dpi=160)
    plot_im(im, title, ax)
    fig.savefig(directory / f"{name}.png")


def run_():
    print(f"Processing image {im_idx}")

    im_directory = save_dir / f"im_{im_idx:02d}"
    im_directory.mkdir(exist_ok=True)

    clean_im, blurred_im = test_ds[im_idx]

    if args.noise_path is None:
        noises = [torch.randn_like(clean_im) * level for level in noise_levels]
    else:
        noises = torch.load(Path(args.noise_path) / f"im_{im_idx:02d}" / "noises.pth")
    torch.save(noises, im_directory / "noises.pth")

    save_i = 0
    save_im(clean_im.squeeze(), "x", f"{save_i:02d}_clean", im_directory)
    torch.save(clean_im, im_directory / "clean_im.pth")
    save_image(clean_im, im_directory / "clean_im.png")
    save_i += 1
    save_im(
        blurred_im.squeeze(),
        f"F(x)\nPSNR={PSNR(clean_im, blurred_im):.2f}\nSSIM={SSIM(clean_im.squeeze(), blurred_im.squeeze()):.2f}",
        f"{save_i:02d}_degraded",
        im_directory,
    )
    torch.save(blurred_im, im_directory / "blurred_im.pth")
    save_image(blurred_im, im_directory / "blurred_im.png")

    for i, noise in enumerate(noises):
        noisy = blurred_im + noise
        save_im(
            noisy.squeeze(),
            rf"F(x) + $\sigma$={noise_levels[i]:.3f}"
            + f"\nPSNR={PSNR(clean_im, noisy):.2f}"
            + f"\nSSIM={SSIM(clean_im.squeeze(), noisy.squeeze()):.2f}",
            f"{save_i:02d}_noisy_{noise_levels[i]:.3f}",
            im_directory,
        )
        torch.save(noisy, im_directory / f"noisy_{noise_levels[i]:.3f}.pth")
        save_image(noisy, im_directory / f"noisy_{noise_levels[i]:.3f}.png")
        save_i += 1

        def restore_(l):
            global save_i
            start = time.time()
            output, _ = restore_image(
                blurred_im + noise,
                model,
                args.device,
                lambda_=l,
                n_iters=args.n_iters,
                use_indicator_fn=True,
                use_least_squares=(args.use_least_squares),
            )

            print(f"Processing time: {time.time() - start}s")
            save_im(
                output.squeeze(),
                r"$\bar{x} \in A(\bar{x}) - y + \lambda \nabla ||\bar{x}||_{TV}$"
                + "\n"
                + rf"$\lambda$={l:.4f}"
                + f"\nPSNR={PSNR(clean_im.squeeze(), output.squeeze()):.2f}"
                + f"\nSSIM={SSIM(clean_im.squeeze(), output.squeeze()):.2f}",
                f"{save_i:02d}_restored_{noise_levels[i]:.3f}_{l:.4f}",
                im_directory,
            )
            torch.save(
                output,
                im_directory / f"restored_{noise_levels[i]:.3f}_{l:.4f}.pth",
            )
            save_image(
                output, im_directory / f"restored_{noise_levels[i]:.3f}_{l:.4f}.png"
            )
            save_i += 1
            return PSNR(clean_im.squeeze(), output.squeeze())

        if not args.use_optuna:
            for l in lambdas:
                restore_(l)
        else:

            def objective(trials):
                l = trials.suggest_float("l", args.lambda_min, args.lambda_max)
                return restore_(l)

            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(),
            )
            study.optimize(objective, n_trials=args.lambda_tries)

            print(f"best_params: {study.best_params}, best_value: {study.best_value}")
            # Save best param in txt file in dir
            with open(
                im_directory / f"best_params_{noise_levels[i]:.3f}.txt", "w"
            ) as f:
                f.write(f"{study.best_params}")

            pruned_trials = study.get_trials(states=[optuna.trial.TrialState.PRUNED])
            complete_trials = study.get_trials(
                states=[optuna.trial.TrialState.COMPLETE]
            )
            print("# Pruned trials: ", len(pruned_trials))
            print("# Complete trials: ", len(complete_trials))
            trial = study.best_trial
            print("Best Score: ", trial.value)
            print("Best Params: ")
            for key, value in trial.params.items():
                print("  {}: {}".format(key, value))


if __name__ == "__main__":
    args = parser.parse_args()

    noise_levels = args.noise_levels
    lambdas = 10 ** torch.linspace(args.lambda_min, args.lambda_max, args.lambda_tries)
    print(f"Noise levels: {noise_levels}")
    print(f"Lambdas: {lambdas}")

    save_dir = Path("results/restoration") / args.save_folder_name
    save_dir.mkdir(exist_ok=True, parents=True)
    print(f"Saving to: {save_dir}")

    blurs = degrad_blur_parameters[args.blur_kernels]
    print(f"Blur used: {blurs}")
    # args.n_images=25
    args.noise_path = None
    _, _, test_ds, _ = get_dataset(
        dataset_name=args.dataset,
        test_dataset_name=args.dataset,
        crop_size=args.image_size,
        batch_size=1,
        blur_parameters=blurs,
        n_images=args.n_images,
        blur_app_type=BlurApplication(args.blur_mode),
        random_crop=False,
        colorized=args.colorized,
        return_test_only=True,
        no_crop=True if args.image_size == -1 else False,
        use_hard_saturation=args.use_hard_saturation,
    )

    if args.model_path is None:
        if args.model_type == "linear":
            print("Using linearized model")
            model = LinearModel(
                blurs,
                args.blur_mode,
                args.use_least_squares,
                use_hard_tanh=args.use_hard_saturation,
            )
        elif args.model_type == "nonlinear":
            print("Using true model")
            model = NonLinearModel(
                blurs,
                args.blur_mode,
                args.use_least_squares,
                use_hard_tanh=args.use_hard_saturation,
            )
        else:
            raise ValueError(f"Unknown model type {args.model_type}")
    else:
        print(f"Loading model from {args.model_path}")
        model = LoadModel.from_file(args.model_path)
        if isinstance(model, SingleFilterConvolutionProjected):
            print("Using the learned linear model")
            model.blur_para.requires_grad = False
            model.use_symmetric_form = (
                args.use_least_squares
            )  # This is only valid for this specific case..
        model.to(args.device)

    for im_idx in range(args.n_images):
        # im_idx = 24
        print(f"Processing image {im_idx}")

        # Create directory
        im_directory = save_dir / f"im_{im_idx:02d}"
        im_directory.mkdir(exist_ok=True)

        # Load image
        clean_im, blurred_im = test_ds[im_idx]
        if args.image_size == -1:
            clean_im = clean_im[
                ..., 1:, 1:
            ]  # Small hack because Unet only takes pair image size
            blurred_im = blurred_im[..., 1:, 1:]

        # Load noise if given else create it. Save the noise
        if args.noise_path is None:
            noises = [torch.randn_like(clean_im) * level for level in noise_levels]
        else:
            noises = torch.load(
                Path(args.noise_path) / f"im_{im_idx:02d}" / "noises.pth"
            )
        torch.save(noises, im_directory / "noises.pth")

        # Save clean and blurry image
        save_i = 0
        save_im(clean_im.squeeze(), "x", f"{save_i:02d}_clean", im_directory)
        torch.save(clean_im, im_directory / "clean_im.pth")
        save_image(clean_im, im_directory / "clean_im.png")
        save_i += 1
        save_im(
            blurred_im.squeeze(),
            f"F(x)\nPSNR={PSNR(clean_im, blurred_im):.2f}\nSSIM={SSIM(clean_im.squeeze(), blurred_im.squeeze()):.2f}",
            f"{save_i:02d}_degraded",
            im_directory,
        )
        torch.save(blurred_im, im_directory / "blurred_im.pth")
        save_image(blurred_im, im_directory / "blurred_im.png")

        # Loop on noise levels
        for i, noise in enumerate(noises):
            print(f"Processing noise level {noise_levels[i]}")
            noisy = blurred_im + noise  # Add noise then save it
            save_im(
                noisy.squeeze(),
                rf"F(x) + $\sigma$={noise_levels[i]:.3f}"
                + f"\nPSNR={PSNR(clean_im, noisy):.2f}"
                + f"\nSSIM={SSIM(clean_im.squeeze(), noisy.squeeze()):.2f}",
                f"{save_i:02d}_noisy_{noise_levels[i]:.3f}",
                im_directory,
            )
            torch.save(noisy, im_directory / f"noisy_{noise_levels[i]:.3f}.pth")
            save_image(noisy, im_directory / f"noisy_{noise_levels[i]:.3f}.png")
            save_i += 1

            def restore_(l, save_i):
                print(f"Processing lambda {l}")
                start = time.time()
                output, metrics = restore_image(
                    blurred_im + noise,
                    model,
                    args.device,
                    lambda_=l,
                    n_iters=args.n_iters,
                    use_indicator_fn=True,
                    use_least_squares=(args.use_least_squares),
                )

                print(
                    f"Processing time: {time.time() - start}s. Saved image in {im_directory}"
                )
                save_im(
                    output.squeeze(),
                    r"$\bar{x} \in A(\bar{x}) - y + \lambda \nabla ||\bar{x}||_{TV}$"
                    + "\n"
                    + rf"$\lambda$={l}"
                    + f"\nPSNR={PSNR(clean_im.squeeze(), output.squeeze()):.2f}"
                    + f"\nSSIM={SSIM(clean_im.squeeze(), output.squeeze()):.2f}",
                    f"{save_i:02d}_restored_{noise_levels[i]:.3f}_{l:.4f}",
                    im_directory,
                )
                torch.save(
                    output,
                    im_directory / f"restored_{noise_levels[i]:.3f}_{l:.4f}.pth",
                )
                save_image(
                    output, im_directory / f"restored_{noise_levels[i]:.3f}_{l:.4f}.png"
                )
                pickle.dump(
                    metrics,
                    open(
                        im_directory / f"metrics_{noise_levels[i]:.3f}_{l:.4f}.pkl",
                        "wb",
                    ),
                )
                return PSNR(clean_im.squeeze(), output.squeeze())

            if not args.use_optuna:
                if not args.parallel:
                    for l in lambdas:
                        restore_(l, save_i=save_i)
                        save_i += 1
                else:
                    from joblib import Parallel, delayed

                    print(f"Using {args.n_jobs} jobs")
                    Parallel(n_jobs=args.n_jobs)(
                        delayed(restore_)(l, save_i + k) for k, l in enumerate(lambdas)
                    )
                    save_i += len(lambdas)
            else:

                def objective(trials):
                    l = trials.suggest_float("l", args.lambda_min, args.lambda_max)
                    return restore_(l, save_i=save_i)

                study = optuna.create_study(
                    direction="maximize",
                    sampler=optuna.samplers.TPESampler(),
                )
                study.optimize(objective, n_trials=args.lambda_tries)

                print(
                    f"best_params: {study.best_params}, best_value: {study.best_value}"
                )
                # Save best param in txt file in dir
                with open(
                    im_directory / f"best_params_{noise_levels[i]:.3f}.txt", "w"
                ) as f:
                    f.write(f"{study.best_params}")

                pruned_trials = study.get_trials(
                    states=[optuna.trial.TrialState.PRUNED]
                )
                complete_trials = study.get_trials(
                    states=[optuna.trial.TrialState.COMPLETE]
                )
                print("# Pruned trials: ", len(pruned_trials))
                print("# Complete trials: ", len(complete_trials))
                trial = study.best_trial
                print("Best Score: ", trial.value)
                print("Best Params: ")
                for key, value in trial.params.items():
                    print("  {}: {}".format(key, value))
        # break
