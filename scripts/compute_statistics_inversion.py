import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm
from PIL import Image
from toolbox.metrics import PSNR, SSIM, MetricsDictionary, mean_absolute_error

from restoration.solver_helper_functions import InverseResult

parser = argparse.ArgumentParser()
parser.add_argument("--results_dirs", type=str, nargs="+", required=True)
parser.add_argument("--save_csv", type=str, required=False, default=None)

if __name__ == "__main__":
    """
    We evaluated each model according to MAE and PSNR
    Need to compute 5 metrics per model:
    - MAE(y_model, y)
    - MAE(x_model, x)
    - MAE(x_model_star, x)
    - PSNR(x_model, x)
    - PSNR(x_model_star, x)
    """

    args = parser.parse_args()

    # Basic checks
    for results_dir in args.results_dirs:
        assert Path(results_dir).exists(), f"{results_dir} does not exist"
        assert Path(results_dir).is_dir(), f"{results_dir} is not a directory"

    df_data = {}
    for results_dir in args.results_dirs:
        print(f"Computing statistics for {results_dir}")
        metrics = MetricsDictionary()
        for im_dir in tqdm.tqdm(list(Path(results_dir).iterdir())):
            if not im_dir.is_dir():
                continue
            im_res = InverseResult.load(im_dir)
            metrics.add(
                {
                    "PSNR_x_xstar": PSNR(im_res.x, im_res.x_star),
                    "MAE_x_xstar": mean_absolute_error(im_res.x, im_res.x_star),
                    "SSIM_x_xstar": SSIM(im_res.x, im_res.x_star),
                    "PSNR_x_xhat": PSNR(im_res.x, im_res.x_hat),
                    "MAE_x_xhat": mean_absolute_error(im_res.x, im_res.x_hat),
                    "SSIM_x_xhat": SSIM(im_res.x, im_res.x_hat),
                    "PSNR_x_y": PSNR(im_res.x, im_res.y),
                }
            )
        df_data[results_dir] = {}
        for k, v in metrics.get_all().items():
            df_data[results_dir][k] = f"{np.mean(v)} (\\pm {np.std(v)})"
        print(results_dir)
        print(df_data[results_dir])
    df_data = pd.DataFrame(df_data).T
    print(df_data.to_latex())

    if args.save_csv is not None:
        df_data.to_csv(args.save_csv)
