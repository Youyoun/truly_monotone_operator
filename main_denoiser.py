### Train a monotone denoiser this time

import argparse
import datetime
from pathlib import Path
from typing import Callable, Tuple, Union

import numpy as np
import pandas as pd
import torch
from itables import to_html_datatable
from toolbox.jacobian import (
    get_lambda_min_or_max_poweriter,
    get_min_max_ev_neuralnet_fulljacobian,
)
from toolbox.metrics import AverageMetricsDictionary, MetricsDictionary
from toolbox.models import Activation, RUnet, get_model
from toolbox.models.pix2pix import get_discriminator
from toolbox.utils import count_parameters
from torchview import draw_graph

from dataset import BlurApplication, degrad_blur_parameters, get_noisy_dataset
from logger import get_module_logger
from templating import get_template_html
from training_functions import Training
from utils import create_log_dir, log_example_image, log_result

HTML_MODEL = "/home/youyoun/Projects/monotone_operator/SimpleMonotoneOperators/model.html"
TEST_POWER_ITER_N = 300

parser = argparse.ArgumentParser(
    "Run the training of an operator for a certain task (saturation in this case)"
)
parser.add_argument(
    "--experiment_name",
    type=str,
    required=True,
    help="Experiment name (prefix of save folder)",
)
parser.add_argument("--save_folder", type=str, default="runs", help="SaveFolder")

parser.add_argument(
    "--dataset",
    type=str,
    choices=["bsd", "imagenet", "hazy"],
    required=True,
    help="Dataset to use",
)
parser.add_argument(
    "--test_dataset",
    type=str,
    choices=["bsd", "imagenet", "hazy"],
    required=True,
    help="Dataset to use",
)
parser.add_argument("--colorized", action="store_true", help="Use colorized version of dataset")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument(
    "--n_images", type=int, default=0, help="Number of images to load. 0 means all"
)
parser.add_argument(
    "--im_size", type=int, default=64, help="The size that will be cropped in images"
)
parser.add_argument("--noise_std", type=float, default=0.0, help="Noise to add to dataset")
parser.add_argument("--random_crop", action="store_true", help="Whether to random crop")
parser.add_argument(
    "--monotony_crop",
    type=int,
    default=0,
    help="Crop the image to this size for monotony. 0 means no crop",
)

parser.add_argument(
    "--input_channels",
    type=int,
    choices=[1, 3],
    default=1,
    help="Number of input channels. 1 for grayscale, 3 for RGB",
)
parser.add_argument(
    "--output_channels",
    type=int,
    choices=[1, 3],
    default=1,
    help="Number of output channels. 1 for grayscale, 3 for RGB",
)
parser.add_argument(
    "--conv_features", type=int, default=16, help="Number of features in middle convs"
)
parser.add_argument("--depth", type=int, default=10, help="Number of convolution layers")
parser.add_argument(
    "--mid_activation",
    type=str,
    choices=[a.name for a in Activation],
    default=Activation.LeakyReLU.name,
    help="Activation of mid layers",
)
parser.add_argument(
    "--last_activation",
    type=str,
    choices=[a.name for a in Activation],
    default=Activation.Identity.name,
    help="Activation of last layer",
)
parser.add_argument(
    "--load_model",
    type=str,
    required=False,
    default=None,
    help="Initialize model with another model",
)
parser.add_argument("--deactivate_gan", action="store_true", help="Do not use GAN learning")
parser.add_argument("--lambda_l1", type=float, default=100, help="Weight for L1 in loss")
parser.add_argument("--learning_rate", type=float, default=0.0002, help="Learning rate")
parser.add_argument("--use_monotony", action="store_true", help="Use monotony penalization")
parser.add_argument(
    "--lambda_monotony", type=float, default=10, help="Weight for Monotony Penalization"
)
parser.add_argument(
    "--eps_monotony", type=float, default=0.05, help="Epsilon for Monotony Penalization"
)
parser.add_argument(
    "--power_iter_tol", type=float, default=1e-5, help="Stop tolerance for power method"
)
parser.add_argument(
    "--max_iter_power_method",
    type=int,
    default=160,
    help="Maximum number of iterations for the power method",
)
parser.add_argument(
    "--lambda_increase_factor",
    type=float,
    default=1,
    help="Increase the factor of monotony lambda",
)
parser.add_argument("--epochs", type=int, required=True, help="Number of epochs for training")
parser.add_argument(
    "--log_every_n_epochs",
    type=int,
    default=10,
    help="every n epoch compute predicted image",
)
parser.add_argument(
    "--device",
    type=str,
    default="cpu",
    choices=["cpu", "cuda"],
    help="Run training on which device",
)
parser.add_argument(
    "--loss",
    type=str,
    default="l1",
    choices=["l1", "l2", "pieapp"],
    help="Which loss to use for training",
)
parser.add_argument(
    "--model_type",
    type=str,
    default="cnn",
    choices=["cnn", "unet", "rrdbnet"],
    help="Which model to use for training",
)


def run_model_test(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    test_save_dir: Union[Path, None],
):
    l1_losses = []
    l2_losses = []
    with torch.no_grad():
        for batch_id, (target, input_) in enumerate(dataloader):
            preds = model(input_.to(device)).cpu()
            losses = [x.item() for x in list(torch.abs(preds - target).mean(dim=[1, 2, 3]))]
            l1_losses.extend(losses)
            losses = [
                x.item() for x in list(torch.square(preds - target).mean(dim=[1, 2, 3]).sqrt())
            ]
            l2_losses.extend(losses)
            for i in range(len(input_)):
                sample_idx = batch_id * len(input_) + i
                if test_save_dir is not None:
                    log_result(
                        input_[i].squeeze(),
                        target[i].squeeze(),
                        preds[i].squeeze(),
                        sample_idx,
                        test_save_dir / f"test_{sample_idx:>03}.png",
                    )
    return l1_losses, l2_losses


def main(opt: argparse.Namespace):
    # CREATE SAVE DIR
    today_ = datetime.datetime.today().strftime("%Y%m%d_%H%M%S")
    BASE_DIR = opt.save_folder
    save_dir = Path(f"{BASE_DIR}/{'monotone' if opt.use_monotony else 'normal'}/run_{today_}")
    create_log_dir(save_dir)
    opt.run_dir = save_dir
    opt.mid_activation = Activation[opt.mid_activation]
    opt.last_activation = Activation[opt.last_activation]

    # CREATE LOGGER
    logger = get_module_logger(__name__, save_dir)
    logger.info(f"Save directory is: {save_dir}")
    logger.info(opt)

    # LOAD DATASETS
    if opt.noise_std == 0:
        raise ValueError("Noise std must be > 0")
    logger.info(f"Using noise with std={opt.noise_std}")

    ds, dl, test_ds, test_dl = get_noisy_dataset(
        dataset_name=opt.dataset,
        test_dataset_name=opt.test_dataset,
        gaussian_mean=0.0,
        gaussian_std=opt.noise_std,
        crop_size=opt.im_size,
        batch_size=opt.batch_size,
        n_images=opt.n_images,
        random_crop=opt.random_crop,
        colorized=opt.colorized,
    )

    # LOG AN EXAMPLE IMAGE
    im_file_path = save_dir.absolute() / "images/example.png"
    idx = 0
    truth, input_ = ds[idx]
    log_example_image(input_.squeeze(), truth.squeeze(), im_file_path)
    logger.info(f"Saved example in file://{im_file_path}")

    # CREATE NN MODELS
    if opt.model_type == "cnn":
        generator = get_model(
            opt.input_channels,
            opt.output_channels,
            opt.conv_features,
            opt.depth,
            opt.last_activation,
            opt.mid_activation,
        ).to(opt.device)
    elif opt.model_type == "unet":
        runet_depth = opt.depth
        filters_ = [opt.conv_features // (2**i) for i in reversed(range(runet_depth))]
        logger.debug(f"Using feature maps for unet: {filters_}")
        generator = RUnet(opt.input_channels, filters_)
    elif opt.model_type == "rrdbnet":
        from toolbox.models import RRDBNet

        generator = RRDBNet(
            opt.input_channels,
            opt.output_channels,
            opt.conv_features,
            opt.depth,
        ).to(opt.device)
    else:
        raise ValueError(f"Unknown model type: {opt.model_type}")
    logger.info(f"Number of parameters: {count_parameters(generator)}")

    if opt.load_model is not None:
        generator.load_state_dict(torch.load(opt.load_model))
        logger.info(f"Loaded model from {opt.load_model}")
        l1_losses, l2_losses = run_model_test(generator, test_dl, opt.device, None)
        logger.info(f"Avg Loss on Test set: {np.mean(l1_losses):.4f} (+- {np.std(l1_losses):.4f})")

    model_graph = draw_graph(
        generator,
        input_size=(1, *input_.shape),
        expand_nested=True,
        graph_name="Operator",
        graph_dir=str(save_dir),
        device=opt.device,
    )
    model_graph.visual_graph.render(save_dir / "Operator", format="png")
    logger.info(generator)

    if opt.deactivate_gan or opt.use_monotony:
        logger.info("No discriminator used since GAN are disabled.")
        discriminator = None
    else:
        discriminator = get_discriminator(
            opt.input_channels + opt.output_channels,
            opt.conv_features,
            "basic",
            norm="none",
        ).to(opt.device)
        model_graph = draw_graph(
            discriminator,
            input_size=(1, 2, *input_.shape[1:]),
            expand_nested=True,
            graph_name="Operator",
            graph_dir=str(save_dir),
            device=opt.device,
        )
        model_graph.visual_graph.render(save_dir / "Discriminator", format="png")
        logger.info(discriminator)

    # CREATE TRAINER AND RUN TRAINING
    trainer = Training(generator, discriminator, opt)
    global_metrics = MetricsDictionary()
    for e in range(opt.epochs):
        if opt.use_monotony:
            opt.lambda_monotony += opt.lambda_increase_factor
            logger.debug(f"Lambda monotony = {opt.lambda_monotony}")
        metrics = AverageMetricsDictionary()
        for i, (real_B, real_A) in enumerate(dl):
            real_A = real_A.to(opt.device)
            real_B = real_B.to(opt.device)
            fake_B = generator(real_A)
            l_d, l_g, l_l1 = trainer.optimall(
                real_A,
                real_B,
                fake_B,
            )
            if opt.use_monotony:
                metrics.add({"Mon": l_g, "L1": l_l1})
            elif opt.deactivate_gan:
                metrics.add({"L1": l_l1})
            else:
                metrics.add({"G": l_g, "D": l_d, "L1": l_l1})
            logger.debug(
                "Epoch: %d, Batch: %d, D: %.3f, G: %.3f, L1: %.3f" % (e, i, l_d, l_g, l_l1)
            )
        if opt.use_monotony:
            logger.info(f"Epoch {e}: Monotony: {metrics['Mon']:.3f} L1: {metrics['L1']:.3f}")
        elif opt.deactivate_gan:
            logger.info(f"Epoch {e}: L1: {metrics['L1']:.3f}")
        else:
            logger.info(
                f"Epoch {e}: Gloss: {metrics['G']:.3f} Dloss: {metrics['D']:.3f} L1: {metrics['L1']:.3f}"
            )
        global_metrics.add(metrics.get_all())

        if e % opt.log_every_n_epochs == 0 or e == opt.epochs - 1:  # Log the training images
            idx = np.random.randint(0, len(ds))
            truth, input_ = ds[idx]
            with torch.no_grad():
                pred = generator(input_.to(opt.device).unsqueeze(0)).squeeze().cpu()
            save_path = save_dir.absolute() / f"images/Epoch_{e:>03}_Idx_{idx}.png"
            log_result(input_.squeeze(), truth.squeeze(), pred, idx, save_path)
            logger.debug(f"Prediction max,min: {pred.min():.3f}, {pred.max():.3f}")
            logger.info(f"Saved example in file://{save_path}")

            # SAVE MODELS
            torch.save(
                generator.state_dict(),
                save_dir / f"G_d{opt.depth}_e{e}.pth",
            )
            if discriminator is not None:
                torch.save(discriminator.state_dict(), save_dir / f"D_d{opt.depth}_e{e}.pth")
            logger.info(f"Saved models in: file://{save_dir.absolute()}")

    # SAVE MODELS
    torch.save(generator.state_dict(), save_dir / f"G_d{opt.depth}_e{opt.epochs}.pth")
    if discriminator is not None:
        torch.save(discriminator.state_dict(), save_dir / f"D_d{opt.depth}_e{opt.epochs}.pth")

    torch.save(
        {
            "global_metrics": global_metrics.get_all(),
            "opt": vars(opt),
            "state_dict": generator.state_dict(),
        },
        save_dir / "last_epoch.pth",
    )
    logger.info(f"Saved models in: file://{save_dir.absolute()}")

    # # Save curves
    # import matplotlib.pyplot as plt

    # for metric, values in global_metrics.get_all().items():
    #     fig, ax = plt.subplots()
    #     ax.plot(values)
    #     ax.set_title(f"Metrics: {metric}")
    #     ax.set_xlabel("Epochs")
    #     ax.set_ylabel(metric)
    #     fig.savefig(save_dir / f"{metric}.png")

    # RUN MODEL ON TEST SET
    test_save_dir = save_dir / "test"
    test_save_dir.mkdir()
    l1_losses, l2_losses = run_model_test(generator, test_dl, opt.device, test_save_dir)
    logger.info(f"Avg Loss on Test set: {np.mean(l1_losses):.4f} (+- {np.std(l1_losses):.4f})")

    # EVALUATE JACOBIAN EVS
    generator.to("cpu")
    logger.info("Evaluating the jacobian")

    def get_min_max_evs(
        g: torch.nn.Module, x: torch.Tensor, opt: argparse.Namespace
    ) -> Tuple[float, float]:
        if (
            opt.im_size > 64
            or opt.depth > 4
            or count_parameters(g) > 4e6
            or opt.model_type == "rrdbnet"
        ):
            logger.info("Using power iter to evaluate jacobian (size of im too big)")
            max_ev = (
                get_lambda_min_or_max_poweriter(
                    g, x, alpha=5, is_eval=False, biggest=True, n_iter=TEST_POWER_ITER_N
                )
                .detach()
                .item()
            )
            min_ev = (
                get_lambda_min_or_max_poweriter(
                    g,
                    x,
                    alpha=max_ev,
                    is_eval=False,
                    biggest=False,
                    n_iter=TEST_POWER_ITER_N,
                )
                .detach()
                .item()
            )
        else:
            logger.info("Using full jacobian to evaluate jacobian")
            min_ev, max_ev = get_min_max_ev_neuralnet_fulljacobian(g, x)
        return min_ev, max_ev

    ev_mins, ev_maxs = [], []
    for idx in range(len(test_ds)):
        try:
            min_, max_ = get_min_max_evs(generator, test_ds[idx][0].unsqueeze(0), opt)
            logger.debug((min_, max_))
            ev_mins.append(min_)
            ev_maxs.append(max_)
        except (RuntimeError, ValueError) as e:
            logger.warning(f"Error while evaluating jacobian for idx {idx}: {e}")
    performance = {
        "TestL1": np.mean(l1_losses),
        "TestL2": np.mean(l2_losses),
        "Lambda_min": np.min(ev_mins),
        "Lambda_min_mean": np.mean(ev_mins),
        "Lambda_max": np.max(ev_maxs),
        "Lambda_max_mean": np.mean(ev_maxs),
    }

    # GENERATE HTML RESULTS FILE
    results_html = save_dir / "index.html"
    if opt.deactivate_gan or opt.use_monotony:
        model_paths = {"Generator": "Operator.png"}
    else:
        model_paths = {
            "Generator": "Operator.png",
            "Discriminator": "Discriminator.png",
        }
    template = get_template_html(
        run_name=save_dir.name,
        args=vars(opt),
        logfile=save_dir / "logs.log",
        performance=performance,
        train_results=global_metrics,
        images_dir=save_dir / "images",
        ev_results=(ev_mins, ev_maxs),
        test_results_paths=[f"test/{path.name}" for path in test_save_dir.iterdir()],
        model_paths=model_paths,
    )
    with open(results_html, "w") as f:
        f.write(template)

    # Add the losses and path to experiment html file.
    table_file = Path(save_dir.parts[0]) / "table.csv"
    index_file = Path(save_dir.parts[0]) / "index.html"
    experiment_results = pd.DataFrame(
        [
            {
                **vars(opt),
                **performance,
                "ExtensiveResults": f'<a href="file://{results_html.absolute()}">link</a>',
                "ModelPath": save_dir / "last_epoch.pth",
            }
        ]
    )
    if table_file.exists():
        experiments = pd.read_csv(table_file, index_col=0)
        experiment_results = experiment_results.append(experiments)
    experiment_results.to_csv(table_file)
    html_table = to_html_datatable(df=experiment_results)
    with open(index_file, "w") as f:
        f.write(html_table)
    logger.info(f"Path to index: file://{index_file.absolute()}")


if __name__ == "__main__":
    opt = parser.parse_args()
    main(opt)
