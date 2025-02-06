import shlex
import shutil
import subprocess
from pathlib import Path

DATASET = "bsd"
EPOCHS = 200
DEPTH = 4
IMSIZE = 256
CONV_FEATURES = 256
INPUT_C = 3
MODEL = "unet"  # Set to "unet" or "cnn" or "linear"
LOSS = "l2"  # Set to "l2" or "l1"
BLUR_MODE = 2
LR = 0.0002
EPS = 0.01
MONTONY_CROP = 64
LAMBDA_MON = 0.1
LAMBDA_INCREASE = 0.01
MAXPOWER = 150

linear_models_paths = {
    "mvt1_small": {
        0.0: {
            "hard_tanh": "/home/youyoun/Projects/MonotoneInverse/runs_20231222_goodinit/runs_mvt1_small_tanh/color/sl/mvt1_small/runs_linear_1_l1/0.0/hard_tanh/normal/run_20231222_134503/last_epoch.pth",
            "tanh": "/home/youyoun/Projects/MonotoneInverse/runs_20231222_goodinit/runs_mvt1_small_tanh/color/sl/mvt1_small/runs_linear_1_l1/0.0/tanh/normal/run_20231222_133705/last_epoch.pth",
        },
        0.01: {
            "hard_tanh": "/home/youyoun/Projects/MonotoneInverse/runs_20231222_goodinit/runs_mvt1_small_tanh/color/sl/mvt1_small/runs_linear_1_l1/0.01/hard_tanh/normal/run_20231222_140123/last_epoch.pth",
            "tanh": "/home/youyoun/Projects/MonotoneInverse/runs_20231222_goodinit/runs_mvt1_small_tanh/color/sl/mvt1_small/runs_linear_1_l1/0.01/tanh/normal/run_20231222_135319/last_epoch.pth",
        },
    },
    "mvt5_small": {
        0.0: {
            "tanh": "/home/youyoun/Projects/MonotoneInverse/runs_20231222_goodinit/runs_mvt5_small_tanh/color/sl/mvt5_small/runs_linear_1_l1/0.0/tanh/normal/run_20231222_140952/last_epoch.pth",
            "hard_tanh": "/home/youyoun/Projects/MonotoneInverse/runs_20231222_goodinit/runs_mvt5_small_tanh/color/sl/mvt5_small/runs_linear_1_l1/0.0/hard_tanh/normal/run_20231222_141824/last_epoch.pth",
        },
        0.01: {
            "hard_tanh": "/home/youyoun/Projects/MonotoneInverse/runs_20231222_goodinit/runs_mvt5_small_tanh/color/sl/mvt5_small/runs_linear_1_l1/0.01/hard_tanh/normal/run_20231222_143530/last_epoch.pth",
            "tanh": "/home/youyoun/Projects/MonotoneInverse/runs_20231222_goodinit/runs_mvt5_small_tanh/color/sl/mvt5_small/runs_linear_1_l1/0.01/tanh/normal/run_20231222_142701/last_epoch.pth",
        },
    },
}

# for blur in ["mvt1_small", "mvt5_small"]:
for blur in ["mvt5_small"]:
    # Set up the save directory
    SAVE_DIR = Path(
        f"runs_pesquet_operator_eps05/runs_{blur}_tanh/color/sl/{blur}/runs_{MODEL}_{CONV_FEATURES}_{LOSS}"
    )
    print(f"Saving to {SAVE_DIR}")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(__file__, SAVE_DIR / "script.py")

    # for NOISE in [0.0, 0.01]:
    for NOISE in [0.0]:
        EXPERIMENT_NAME = f"Unet_Noise{NOISE}_Model{MODEL}_NConv{CONV_FEATURES}_Loss{LOSS}_depth{DEPTH}_imsize{IMSIZE}_blur{blur}_blurmode{BLUR_MODE}"

        print(f"Running {EXPERIMENT_NAME}")
        for use_hard_tanh in [False, True]:
            if NOISE == 0.0 and not use_hard_tanh and blur == "mvt1_small":
                continue
            # fmt:off
            command = [
                "python", "main.py",
                "--experiment_name", EXPERIMENT_NAME,
                "--dataset", DATASET,
                "--save_folder", f"{SAVE_DIR}/{NOISE}/{'hard_tanh' if use_hard_tanh else 'tanh'}",
                "--device", "cuda",
                "--epochs", str(EPOCHS),
                "--deactivate_gan",
                "--depth", str(DEPTH),
                "--last_activation", "Identity",
                "--learning_rate", str(LR),
                "--conv_features", str(CONV_FEATURES),
                "--noise_std", str(NOISE),
                "--blur_parameters", blur,
                "--random_crop",
                "--test_dataset", DATASET,
                "--degradation_fn", str(BLUR_MODE),
                "--colorized",
                "--input_channels", str(INPUT_C),
                "--output_channels", str(INPUT_C),
                "--model_type", MODEL,
                "--loss", LOSS,
                "--im_size", str(IMSIZE),
                "--log_every_n_epochs", "10",
                "--use_linear_convolution",
                "--linear_model_path", linear_models_paths[blur][NOISE]["hard_tanh" if use_hard_tanh else "tanh"],
                "--use_monotony", 
                "--lambda_increase_factor", str(LAMBDA_INCREASE),
                "--lambda_monotony", str(LAMBDA_MON),
                "--max_iter_power_method", str(MAXPOWER),
                "--monotony_crop", str(MONTONY_CROP),
                "--eps_monotony", str(EPS),
            ]
            # fmt:on
            if use_hard_tanh:
                command.append("--use_hard_saturation")

            command_str = " ".join(shlex.quote(arg) for arg in command)

            # Print the command for clarity
            print(f"Running: {command_str}")

            # Run the command
            subprocess.run(command)
