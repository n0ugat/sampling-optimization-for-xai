#!/bin/bash
#BSUB -q hpc
#BSUB -J attributions_test_eval
#BSUB -n 8
#BSUB -W 00:10
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=500MB]"
#BSUB -o outputs/hpclogs/attributions_test_eval_%J.out
#BSUB -e outputs/hpclogs/attributions_test_eval_%J.err

source ~/miniforge3/etc/profile.d/conda.sh
conda activate freqrise

lscpu
echo "-----------------------------------------"
python main_evaluation.py --dataset synthetic --compute_localization_scores --noise_level 0.0 --synth_sig_len 100 --n_samples 50 --compute_complexity_scores --compute_deletion_scores

