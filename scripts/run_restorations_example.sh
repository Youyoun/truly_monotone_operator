#!/bin/bash
# Run this script from the root directory of the project
# Example: bash scripts/run_restorations_example.sh
# Script will compute the restoration of a blurred image using different trained models and different noise levels
# The script will also compute the restoration of a blurred image using a linear model

dataset=bsd
image_size=256
n_images=10
blur_mode=2
device=cuda
n_iters=2000
colorized= #--colorized
kernels=single

noise=0.0 # Training noise level
type=bw
if [ "$colorized" = "--colorized" ]; then
    type=color
fi
save_folder_name=sl_${type}_${kernels}_${noise} # Results are saved in ./results/restoration/${save_folder_name}

# Path to the trained model
model_path=
python scripts/run_restoration_test.py --dataset $dataset --image_size $image_size --n_images $n_images --blur_kernels $kernels --blur_mode $blur_mode "$colorized" --save_folder_name "$save_folder_name" --n_iters $n_iters --device $device --model_path "$model_path"

noise_path=./results/restoration/${save_folder_name}

# Linear model (no training)
python scripts/run_restoration_test.py --dataset $dataset --image_size $image_size --n_images $n_images --blur_kernels $kernels --blur_mode $blur_mode "$colorized" --save_folder_name $save_folder_name --n_iters $n_iters --device $device --noise_path $noise_path
