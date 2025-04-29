#!/bin/bash

# Define the variables
labeltype="digit" # Options: "digit", "gender"
n_samples=10
freqrise_samples=3000
batch_size=10
num_cells=200
lr=0.0001
alpha=1.0
beta=0.01
decay=0.9 # Options: Interval: [0,1]
reward_fn="pred" # Options: "pred", "saliency"
use_softmax="False"

jobname="freqrise_${labeltype}_${n_samples}_${freqrise_samples}_${batch_size}_${num_cells}_${lr}_${alpha}_${beta}_${decay}_${reward_fn}_${use_softmax}"
cores=8
memory=4000
memory_per_core=$((memory / cores))

# Generate the jobscript.sh
cat <<EOF > jobscript.sh
#!/bin/bash
#BSUB -q hpc
#BSUB -J $jobname
#BSUB -n $cores
#BSUB -W 04:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=${memory_per_core}MB]"
#BSUB -o outputs/hpclogs/${jobname}%J.out
#BSUB -e outputs/hpclogs/${jobname}%J.err

source ~/miniforge3/etc/profile.d/conda.sh
conda activate freqrise

lscpu
echo "-----------------------------------------"
python main_attributions_reinforce.py \\
    --labeltype $labeltype \\
    --n_samples $n_samples \\
    --freqrise_samples $freqrise_samples \\
    --batch_size $batch_size \\
    --num_cells $num_cells \\
    --lr $lr \\
    --alpha $alpha \\
    --beta $beta \\
    --decay $decay \\
    --reward_fn $reward_fn \\
    --use_softmax $use_softmax
echo "-----------------------------------------"
python main_attributions_reinforce.py \\
    --labeltype gender \\
    --n_samples $n_samples \\
    --freqrise_samples $freqrise_samples \\
    --batch_size $batch_size \\
    --num_cells $num_cells \\
    --lr $lr \\
    --alpha $alpha \\
    --beta $beta \\
    --decay $decay \\
    --reward_fn $reward_fn \\
    --use_softmax $use_softmax
echo "-----------------------------------------"
python main_attributions_reinforce.py \\
    --labeltype $labeltype \\
    --n_samples $n_samples \\
    --freqrise_samples 500 \\
    --batch_size $batch_size \\
    --num_cells $num_cells \\
    --lr $lr \\
    --alpha $alpha \\
    --beta $beta \\
    --decay $decay \\
    --reward_fn $reward_fn \\
    --use_softmax $use_softmax
echo "-----------------------------------------"
python main_attributions_reinforce.py \\
    --labeltype $labeltype \\
    --n_samples $n_samples \\
    --freqrise_samples 1000 \\
    --batch_size $batch_size \\
    --num_cells $num_cells \\
    --lr $lr \\
    --alpha $alpha \\
    --beta $beta \\
    --decay $decay \\
    --reward_fn $reward_fn \\
    --use_softmax $use_softmax
echo "-----------------------------------------"
python main_attributions_reinforce.py \\
    --labeltype $labeltype \\
    --n_samples $n_samples \\
    --freqrise_samples 10000 \\
    --batch_size $batch_size \\
    --num_cells $num_cells \\
    --lr $lr \\
    --alpha $alpha \\
    --beta $beta \\
    --decay $decay \\
    --reward_fn $reward_fn \\
    --use_softmax $use_softmax
echo "-----------------------------------------"
python main_attributions_reinforce.py \\
    --labeltype $labeltype \\
    --n_samples $n_samples \\
    --freqrise_samples $freqrise_samples \\
    --batch_size 50 \\
    --num_cells $num_cells \\
    --lr $lr \\
    --alpha $alpha \\
    --beta $beta \\
    --decay $decay \\
    --reward_fn $reward_fn \\
    --use_softmax $use_softmax
echo "-----------------------------------------"
python main_attributions_reinforce.py \\
    --labeltype $labeltype \\
    --n_samples $n_samples \\
    --freqrise_samples $freqrise_samples \\
    --batch_size $batch_size \\
    --num_cells 40 \\
    --lr $lr \\
    --alpha $alpha \\
    --beta $beta \\
    --decay $decay \\
    --reward_fn $reward_fn \\
    --use_softmax $use_softmax
echo "-----------------------------------------"
python main_attributions_reinforce.py \\
    --labeltype $labeltype \\
    --n_samples $n_samples \\
    --freqrise_samples $freqrise_samples \\
    --batch_size $batch_size \\
    --num_cells 1000 \\
    --lr $lr \\
    --alpha $alpha \\
    --beta $beta \\
    --decay $decay \\
    --reward_fn $reward_fn \\
    --use_softmax $use_softmax
echo "-----------------------------------------"
python main_attributions_reinforce.py \\
    --labeltype $labeltype \\
    --n_samples $n_samples \\
    --freqrise_samples $freqrise_samples \\
    --batch_size $batch_size \\
    --num_cells 2000 \\
    --lr $lr \\
    --alpha $alpha \\
    --beta $beta \\
    --decay $decay \\
    --reward_fn $reward_fn \\
    --use_softmax $use_softmax
echo "-----------------------------------------"
python main_attributions_reinforce.py \\
    --labeltype $labeltype \\
    --n_samples $n_samples \\
    --freqrise_samples $freqrise_samples \\
    --batch_size $batch_size \\
    --num_cells $num_cells \\
    --lr 0.001 \\
    --alpha $alpha \\
    --beta $beta \\
    --decay $decay \\
    --reward_fn $reward_fn \\
    --use_softmax $use_softmax
echo "-----------------------------------------"
python main_attributions_reinforce.py \\
    --labeltype $labeltype \\
    --n_samples $n_samples \\
    --freqrise_samples $freqrise_samples \\
    --batch_size $batch_size \\
    --num_cells $num_cells \\
    --lr 0.01 \\
    --alpha $alpha \\
    --beta $beta \\
    --decay $decay \\
    --reward_fn $reward_fn \\
    --use_softmax $use_softmax
echo "-----------------------------------------"
python main_attributions_reinforce.py \\
    --labeltype $labeltype \\
    --n_samples $n_samples \\
    --freqrise_samples $freqrise_samples \\
    --batch_size $batch_size \\
    --num_cells $num_cells \\
    --lr $lr \\
    --alpha $alpha \\
    --beta 0.1 \\
    --decay $decay \\
    --reward_fn $reward_fn \\
    --use_softmax $use_softmax
echo "-----------------------------------------"
python main_attributions_reinforce.py \\
    --labeltype $labeltype \\
    --n_samples $n_samples \\
    --freqrise_samples $freqrise_samples \\
    --batch_size $batch_size \\
    --num_cells $num_cells \\
    --lr $lr \\
    --alpha $alpha \\
    --beta 0.001 \\
    --decay $decay \\
    --reward_fn $reward_fn \\
    --use_softmax $use_softmax
echo "-----------------------------------------"
python main_attributions_reinforce.py \\
    --labeltype $labeltype \\
    --n_samples $n_samples \\
    --freqrise_samples $freqrise_samples \\
    --batch_size $batch_size \\
    --num_cells $num_cells \\
    --lr $lr \\
    --alpha $alpha \\
    --beta 0.5 \\
    --decay $decay \\
    --reward_fn $reward_fn \\
    --use_softmax $use_softmax
echo "-----------------------------------------"
python main_attributions_reinforce.py \\
    --labeltype $labeltype \\
    --n_samples $n_samples \\
    --freqrise_samples $freqrise_samples \\
    --batch_size $batch_size \\
    --num_cells $num_cells \\
    --lr $lr \\
    --alpha $alpha \\
    --beta $beta \\
    --decay 0.99 \\
    --reward_fn $reward_fn \\
    --use_softmax $use_softmax
echo "-----------------------------------------"
python main_attributions_reinforce.py \\
    --labeltype $labeltype \\
    --n_samples $n_samples \\
    --freqrise_samples $freqrise_samples \\
    --batch_size $batch_size \\
    --num_cells $num_cells \\
    --lr $lr \\
    --alpha $alpha \\
    --beta $beta \\
    --decay 0.5 \\
    --reward_fn $reward_fn \\
    --use_softmax $use_softmax
echo "-----------------------------------------"
python main_attributions_reinforce.py \\
    --labeltype $labeltype \\
    --n_samples $n_samples \\
    --freqrise_samples $freqrise_samples \\
    --batch_size $batch_size \\
    --num_cells $num_cells \\
    --lr $lr \\
    --alpha $alpha \\
    --beta $beta \\
    --decay $decay \\
    --reward_fn saliency \\
    --use_softmax $use_softmax
echo "-----------------------------------------"
python main_attributions_reinforce.py \\
    --labeltype $labeltype \\
    --n_samples $n_samples \\
    --freqrise_samples $freqrise_samples \\
    --batch_size $batch_size \\
    --num_cells $num_cells \\
    --lr $lr \\
    --alpha $alpha \\
    --beta $beta \\
    --decay $decay \\
    --reward_fn $reward_fn \\
    --use_softmax True
EOF

# Submit the job
bsub < jobscript.sh

# Clean up
rm jobscript.sh