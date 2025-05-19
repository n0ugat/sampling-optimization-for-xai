#!/bin/bash
#BSUB -q gpuv100
#BSUB -J attributions_test_eval_AudioMNIST_gender
#BSUB -n 8
#BSUB -W 01:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=500MB]"
#BSUB -w ended(25021180)
#BSUB -o outputs/hpclogs/attributions_test_eval_AudioMNIST_gender_%J.out
#BSUB -e outputs/hpclogs/attributions_test_eval_AudioMNIST_gender_%J.err

source ~/miniforge3/etc/profile.d/conda.sh
conda activate freqrise

lscpu
echo "-----------------------------------------"
python main_evaluation.py --dataset AudioMNIST --labeltype gender --n_samples 10 --compute_complexity_scores --compute_deletion_scores

