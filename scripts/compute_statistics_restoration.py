import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import tqdm
from PIL import Image
from toolbox.metrics import PSNR, SSIM, pieapp

# Create logger with file handler and stream handler

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s:%(name)s:%(message)s")
file_handler = logging.FileHandler("restoration_statistics.log")
file_handler.setFormatter(formatter)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)


parser = argparse.ArgumentParser()
parser.add_argument("--results_dir", type=str, nargs="+", required=True)
parser.add_argument("--save_file", type=str, default="results.csv")
parser.add_argument("--by_lambda", action="store_true")
parser.add_argument("--by_image", action="store_true")


if __name__ == "__main__":
    """
    We evaluated a model's ability to restore a degraded image (blur + noise) according to MAE and PSNR
    We use two metrics for each noise level:
    - PSNR
    - SSIM
    """

    args = parser.parse_args()

    results_dirs = [Path(dir_) for dir_ in args.results_dir]
    global_results = {}
    for results_dir in results_dirs:
        logger.info(f"Processing {results_dir}")
        pth_files = list(results_dir.glob("*/restored_*"))
        noise_levels = sorted(set([float(pth.stem.split("_")[1]) for pth in pth_files]))
        logger.info(f"Found noise levels: {noise_levels}")
        if args.by_lambda:
            lambdas = sorted(set([float(pth.stem.split("_")[2]) for pth in pth_files]))
            results = {
                "PSNR": {level: {l: [] for l in lambdas} for level in noise_levels},
                "SSIM": {level: {l: [] for l in lambdas} for level in noise_levels},
                "PIEAPP": {level: {l: [] for l in lambdas} for level in noise_levels},
            }
        elif args.by_image:
            results = {
                "PSNR": {level: [] for level in noise_levels},
                "SSIM": {level: [] for level in noise_levels},
                "PIEAPP": {level: [] for level in noise_levels},
            }
        else:
            results = {
                "PSNR": {level: [] for level in noise_levels},
                "SSIM": {level: [] for level in noise_levels},
                "PIEAPP": {level: [] for level in noise_levels},
            }

        # I have three models, linear, monotone and non monotone
        for im_dir in results_dir.iterdir():
            logger.info(f"Processing {im_dir}")
            x = torch.load(im_dir / "clean_im.pth").numpy().squeeze()

            for noise_level in noise_levels:
                max_metrics = {
                    "PSNR": -float("inf"),
                    "SSIM": 0,
                    "PIEAPP": float("inf"),
                }
                best_im_file = None
                pth_files = list(im_dir.glob(f"restored_{noise_level:.3f}_*.pth"))

                for pth_file in pth_files:
                    x_star = torch.load(pth_file).numpy().squeeze()

                    ssim = SSIM(x, x_star)
                    psnr = PSNR(x, x_star)
                    pieapp_ = 0  # pieapp(x, x_star)

                    if args.by_lambda:
                        lambda_ = float(pth_file.stem.split("_")[2])
                        results["PSNR"][noise_level][lambda_].append(psnr)
                        results["SSIM"][noise_level][lambda_].append(ssim)
                        results["PIEAPP"][noise_level][lambda_].append(pieapp_)
                    else:
                        max_metrics["PSNR"] = max(max_metrics["PSNR"], psnr)
                        max_metrics["SSIM"] = max(max_metrics["SSIM"], ssim)
                        max_metrics["PIEAPP"] = min(max_metrics["PIEAPP"], pieapp_)

                if not args.by_lambda:
                    results["PSNR"][noise_level].append(max_metrics["PSNR"])
                    results["SSIM"][noise_level].append(max_metrics["SSIM"])
                    results["PIEAPP"][noise_level].append(max_metrics["PIEAPP"])

        # Compute the mean and std of the results
        for metric in results:
            for noise_level in results[metric]:
                if args.by_lambda:
                    for lambda_ in results[metric][noise_level]:
                        results[metric][noise_level][lambda_] = {
                            "mean": np.mean(results[metric][noise_level][lambda_]),
                            "std": np.std(results[metric][noise_level][lambda_]),
                        }
                else:
                    results[metric][noise_level] = {
                        "mean": np.mean(results[metric][noise_level]),
                        "std": np.std(results[metric][noise_level]),
                    }

        # Print the results formatted
        for metric in results:
            logger.debug(metric)
            for noise_level in results[metric]:
                if args.by_lambda:
                    logger.debug(f"\t{noise_level:.4f}:")
                    for lambda_ in results[metric][noise_level]:
                        logger.debug(
                            f"\t\t{lambda_:.4f}: {results[metric][noise_level][lambda_]['mean']:.3f} ($\pm$ {results[metric][noise_level][lambda_]['std']:.3f})"
                        )
                else:
                    logger.debug(
                        f"\t{noise_level:.4f}: {results[metric][noise_level]['mean']:.3f} ($\pm$ {results[metric][noise_level]['std']:.3f})"
                    )

        global_results[results_dir] = results

    # Print the global results and save as pandas dataframe
    import pandas as pd

    df = pd.DataFrame()
    for results_dir in global_results:
        for metric in global_results[results_dir]:
            for noise_level in global_results[results_dir][metric]:
                if args.by_lambda:
                    for lambda_ in global_results[results_dir][metric][noise_level]:
                        df = pd.concat(
                            [
                                df,
                                pd.DataFrame(
                                    {
                                        "results_dir": [results_dir],
                                        "metric": [metric],
                                        "noise_level": [noise_level],
                                        "lambda": [lambda_],
                                        "mean": [
                                            global_results[results_dir][metric][
                                                noise_level
                                            ][lambda_]["mean"]
                                        ],
                                        "std": [
                                            global_results[results_dir][metric][
                                                noise_level
                                            ][lambda_]["std"]
                                        ],
                                    }
                                ),
                            ],
                            axis=0,
                            ignore_index=True,
                        )
                else:
                    df = pd.concat(
                        [
                            df,
                            pd.DataFrame(
                                {
                                    "results_dir": [results_dir],
                                    "metric": [metric],
                                    "noise_level": [noise_level],
                                    "mean": [
                                        global_results[results_dir][metric][
                                            noise_level
                                        ]["mean"]
                                    ],
                                    "std": [
                                        global_results[results_dir][metric][
                                            noise_level
                                        ]["std"]
                                    ],
                                }
                            ),
                        ],
                        axis=0,
                        ignore_index=True,
                    )

    df.to_csv(args.save_file)
