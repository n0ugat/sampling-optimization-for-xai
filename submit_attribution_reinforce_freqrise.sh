#!/bin/bash

# Define the variables
labeltype="digit"
n_samples=10
freqrise_samples=3000
batch_size=10
num_cells=200
lr=0.0001
alpha=1.0
beta=0.01
decay=0.9
reward_fn="pred"
use_softmax=false

jobname="freqrise_${labeltype}_${n_samples}_${freqrise_samples}_${batch_size}_${num_cells}_${lr}_${alpha}_${beta}_${decay}_${reward_fn}_${use_softmax}"
cores=8
memory=4000
memory_per_core=500

# Generate the jobscript.sh
cat <<EOF > jobscript.sh
#!/bin/bash
#BSUB -q hpc
#BSUB -J $jobname
#BSUB -n $cores
#BSUB -W 00:15
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
EOF

# Submit the job
bsub < jobscript.sh

# Clean up
rm jobscript.sh