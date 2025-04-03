#!/bin/bash
#BSUB -q gpuv100
#BSUB -J freqrise_gender
#BSUB -n 8
#BSUB -W 02:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=500MB]"
#BSUB -o outputs/freqrise_gender%J.out
#BSUB -e outputs/freqrise_gender%J.err

source ~/miniforge3/etc/profile.d/conda.sh
conda activate freqrise

lscpu
echo "-----------------------------------------"
python main_attributions.py --labeltype gender --n_samples 1000 --freqrise_samples 10000

