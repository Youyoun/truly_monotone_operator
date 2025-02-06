#!/bin/bash
# Run this script from the root directory of the project
# Example: bash scripts/run_train_example.sh

# Set up the save directory
SAVE_DIR=save_dir
mkdir -p "$SAVE_DIR"
cp "$0" "$SAVE_DIR"/

# Set up the parameters
DATASET=bsd
EPOCHS=100
DEPTH=8
CONV_FEATURES=128
BLUR_TYPE=ten # Look in the code for the different types of blur
BLUR_MODE=2   # Look in the code for the different modes of blur
COLORIZED=    # Set to --colorized if we want to train a colorized model
INPUT_C=1
if [ "$COLORIZED" ]; then
    INPUT_C=3
fi

NOISE=0.0

# Run non-monotone
EXPERIMENT_NAME=Noise${NOISE}
python main.py --experiment_name "$EXPERIMENT_NAME" --dataset $DATASET --save_folder "${SAVE_DIR}/${NOISE}" --device cuda --epochs $EPOCHS --depth $DEPTH --last_activation Tanh --learning_rate 0.0002 --conv_features $CONV_FEATURES --deactivate_gan --noise_std "${NOISE}" --blur_parameters $BLUR_TYPE --random_crop --test_dataset $DATASET --degradation_fn $BLUR_MODE "$COLORIZED" --input_channels $INPUT_C --output_channels $INPUT_C

# Run monotone
EXPERIMENT_NAME=MonotoneNoise${NOISE}
LAMBDA_MON=0.0
LAMBDA_INCREASE=1.0
MAXPOWER=150
python main.py --experiment_name "$EXPERIMENT_NAME" --dataset $DATASET --save_folder "${SAVE_DIR}/${NOISE}" --device cuda --epochs $EPOCHS --depth $DEPTH --last_activation Tanh --learning_rate 0.0002 --conv_features $CONV_FEATURES --deactivate_gan --noise_std $NOISE --blur_parameters $BLUR_TYPE --random_crop --use_monotony --lambda_increase_factor $LAMBDA_INCREASE --lambda_monotony $LAMBDA_MON --max_iter_power_method $MAXPOWER --test_dataset $DATASET --degradation_fn $BLUR_MODE "$COLORIZED" --input_channels $INPUT_C --output_channels $INPUT_C
