#!/bin/bash
#BSUB -q hpc
#BSUB -J freqrise_digit_eval
#BSUB -n 8
#BSUB -W 05:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=500MB]"
#BSUB -o outputs/freqrise_digit_eval%J.out
#BSUB -e outputs/freqrise_digit_eval%J.err

source ~/miniforge3/etc/profile.d/conda.sh
conda activate freqrise

lscpu
echo "-----------------------------------------"
python main_evaluation.py --dataset synthetic --compute_localization_scores --labeltype digit --noise_level 0.0 --synth_sig_len 100 --n_samples 10 --compute_complexity_scores --compute_deletion_scores

