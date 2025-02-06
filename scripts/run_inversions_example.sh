#!/bin/bash
# Run this script from the root directory of the project
# Example: bash scripts/run_inversions_example.sh
# Script will compute the inversion of a blurred image using different trained models and different noise levels
# The script will also compute the inversion of a blurred image using a linear model

kernels=single
n_images=0
blur_mode=2
device=cuda
n_iters=2000
image_size=128
colorized= #--colorized

type=bw
if [ "$colorized" = "--colorized" ]; then
    type=color
fi

noise=0.0
save_folder_name=sl_${type}_${kernels}_${noise}_${image_size}

# Path to the trained models
non_mon_model_path=
mon_model_path=
python scripts/run_inversion_results.py --non_mon_model_path "$non_mon_model_path" --mon_model_path "$mon_model_path" --blur_kernels $kernels --n_images $n_images --blur_mode $blur_mode --save_folder_name $save_folder_name --device $device "$colorized" --n_iters $n_iters --image_size $image_size
