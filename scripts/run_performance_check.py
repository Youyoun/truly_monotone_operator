import argparse
import time
from pathlib import Path
import pandas as pd

import matplotlib.pyplot as plt
import torch
from toolbox.metrics import PSNR, SSIM, pieapp, mean_absolute_error
from toolbox.jacobian import MonotonyRegularization, PenalizationMethods

from dataset import degrad_blur_parameters, get_dataset
from degradation import BlurApplication
from restoration.operators import LinearModel
from restoration.plot_functions import plot_im
from restoration.utils import load_model

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
parser.add_argument("--device", type=str, default="cuda", required=True)

parser.add_argument("--model_type", type=str, choices=["cnn", "unet"], required=True)
parser.add_argument("--depth", type=int, default=8)
parser.add_argument("--conv_features", type=int, default=128)


def save_im(im, title, name, directory):
    fig, ax = plt.subplots(dpi=160)
    plot_im(im, title, ax)
    fig.savefig(directory / f"{name}.png")


if __name__ == "__main__":
    args = parser.parse_args()

    save_dir = Path("results/performance") / args.save_folder_name
    save_dir.mkdir(exist_ok=True, parents=True)
    print(f"Saving to: {save_dir}")

    blurs = degrad_blur_parameters[args.blur_kernels]
    print(f"Blur used: {blurs}")

    _, _, test_ds, _ = get_dataset(
        args.dataset,
        args.dataset,
        args.image_size,
        1,
        blurs,
        args.n_images,
        args.blur_mode,
        return_test_only=True,
        colorized=args.colorized,
    )

    if args.model_path is None:
        print("Using linearized model")
        model = LinearModel(blurs, args.blur_mode)
    else:
        print(f"Loading model from {args.model_path}")
        model = load_model(
            args.depth,
            args.conv_features,
            args.model_path,
            in_c=3 if args.colorized else 1,
            out_c=3 if args.colorized else 1,
            model_type=args.model_type,
        )
        model.eval()
        model.to(args.device)

    regul = MonotonyRegularization(
        PenalizationMethods.LANCZOS,
        eps=0.0,
        alpha=0.0,
        max_iter=3000,
        power_iter_tol=1e-5,
        eval_mode=True,
        use_relu_penalization=False,
    )

    PSNRs = []
    SSIMs = []
    MAEs = []
    PieAPPs = []
    Min_Evs_X = []
    Min_Evs_Y = []
    for im_idx in range(len(test_ds)):
        print(f"Processing image {im_idx}")

        clean_im, blurred_im = test_ds[im_idx]

        with torch.no_grad():
            start = time.time()
            predicted_im = model(clean_im.unsqueeze(0).to(args.device)).squeeze(0)
            end = time.time()

        print(f"Time taken: {end - start:.2f}s")
        print(f"Computing EV")
        if args.model_path is None:
            min_ev_x = 0.0
            min_ev_y = 0.0
        else:
            _, min_ev_x = regul(model, clean_im.unsqueeze(0).to(args.device))
            _, min_ev_y = regul(model, predicted_im.unsqueeze(0).to(args.device))
            min_ev_x = min_ev_x.item()
            min_ev_y = min_ev_y.item()

        # Compute metrics
        psnr = PSNR(blurred_im, predicted_im)
        ssim = SSIM(blurred_im, predicted_im)
        pieapp_ = pieapp(blurred_im, predicted_im)
        mae_ = mean_absolute_error(blurred_im, predicted_im)

        print(
            f"{im_idx} - PSNR: {psnr:.2f} dB | SSIM: {ssim:.2f} | PIEAPP: {pieapp_:.2f} | EV: {min_ev_y:.2f} | EV: {min_ev_x:.2f} | MAE: {mae_:.2f}"
        )

        PSNRs.append(psnr)
        SSIMs.append(ssim)
        PieAPPs.append(pieapp_)
        Min_Evs_X.append(min_ev_x)
        Min_Evs_Y.append(min_ev_y)
        MAEs.append(mae_)

    # Save tables
    payload = pd.DataFrame(
        {
            "PSNR": PSNRs,
            "SSIM": SSIMs,
            "PieAPP": PieAPPs,
            "Min EV X": Min_Evs_X,
            "Min EV Y": Min_Evs_Y,
            "MAE": MAEs,
        },
        index=[test_ds.get_image_path(i) for i in range(len(test_ds))],
    )
    payload.to_csv(save_dir / "metrics.csv")
