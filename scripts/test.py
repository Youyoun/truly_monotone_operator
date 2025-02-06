import torch
from pathlib import Path
from tqdm import tqdm
import argparse
from toolbox.metrics import PSNR, SSIM, pieapp
from toolbox.imageOperators import BlurConvolution
import time


def run_(test_ds, im_idx, save_dir, noise_levels, args, model, lambdas):
    print(f"Processing image {im_idx}")

    # Create directory
    im_directory = save_dir / f"im_{im_idx:02d}"
    im_directory.mkdir(exist_ok=True)

    # Load image
    clean_im, blurred_im = test_ds[im_idx]

    # Load noise if given else create it. Save the noise
    if args.noise_path is None:
        noises = [torch.randn_like(clean_im) * level for level in noise_levels]
    else:
        noises = torch.load(Path(args.noise_path) / f"im_{im_idx:02d}" / "noises.pth")
    torch.save(noises, im_directory / "noises.pth")

    # Save clean and blurry image
    save_i = 0
    save_im(clean_im.squeeze(), "x", f"{save_i:02d}_clean", im_directory)
    torch.save(clean_im, im_directory / "clean_im.pth")
    save_i += 1
    save_im(
        blurred_im.squeeze(),
        f"F(x)\nPSNR={PSNR(clean_im, blurred_im):.2f}\nSSIM={SSIM(clean_im.squeeze(), blurred_im.squeeze()):.2f}",
        f"{save_i:02d}_degraded",
        im_directory,
    )
    torch.save(blurred_im, im_directory / "blurred_im.pth")

    # Loop on noise levels
    for i, noise in enumerate(noises):
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
                use_least_squares=(args.model_path is None and args.use_least_squares),
            )

            print(f"Processing time: {time.time() - start}s")
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
            save_i += 1
            return PSNR(clean_im.squeeze(), output.squeeze())

        if not args.use_optuna:
            if not args.parallel:
                for l in lambdas:
                    restore_(l)
            else:
                from joblib import Parallel, delayed

                Parallel(n_jobs=-1)(
                    delayed(restore_)(l) for l in tqdm(lambdas, desc="Parallel")
                )
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
