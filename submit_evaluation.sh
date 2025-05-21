#!/bin/bash
#BSUB -q gpuv100
#BSUB -J im_eval_AudioMNIST_gender
#BSUB -n 8
#BSUB -W 01:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=500MB]"
#BSUB -o outputs/hpclogs/im_eval_AudioMNIST_gender_%J.out
#BSUB -e outputs/hpclogs/im_eval_AudioMNIST_gender_%J.err

source ~/miniforge3/etc/profile.d/conda.sh
conda activate freqrise

lscpu
echo "-----------------------------------------"
# --compute_localization_scores
python main_evaluation.py --dataset AudioMNIST --labeltype gender --synth_sig_len 100 --n_samples 10 --compute_complexity_scores --compute_deletion_scores --incrementing_masks