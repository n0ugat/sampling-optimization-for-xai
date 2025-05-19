#!/bin/bash

# Define the variables
jobname="compute_attributions_test"
merge_jobname="merge_${jobname}"

dataset="synthetic"
output_path="outputs"
n_samples=50

# If using AudioMNIST
labeltype="digit" 

# If using synthetic
noise_level=0.0
synth_sig_len=100

# Generate the jobarray.sh
cat <<EOF > jobarray.sh
#!/bin/bash
#BSUB -q voltash
#BSUB -J ${jobname}[1-11]
#BSUB -n 4
#BSUB -W 00:30
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2GB]"
#BSUB -o outputs/hpclogs/${jobname}_%J/%I.out
#BSUB -e outputs/hpclogs/${jobname}_%J/%I.err

source ~/miniforge3/etc/profile.d/conda.sh
conda activate freqrise

lscpu
echo "-----------------------------------------"
python main_attributions.py \\
    --job_idx $LSB_JOBINDEX \\
    --job_name $jobname \\
    --data_path data/ \\
    --model_path models \\
    --output_path $output_path \\
    --dataset $dataset \\
    --noise_level $noise_level \\
    --synth_sig_len $synth_sig_len \\
    --labeltype $labeltype \\
    --n_samples $n_samples \\
EOF

# Submit the job
bsub < jobarray.sh

# Clean up
rm jobarray.sh


# Generate the merge_outputs.sh
cat <<EOF > merge_outputs.sh
#!/bin/bash
#BSUB -q hpc
#BSUB -J ${merge_jobname}
#BSUB -n 1
#BSUB -W 00:10
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -w ended(${jobname})
#BSUB -o outputs/hpclogs/${merge_jobname}_%J.out
#BSUB -e outputs/hpclogs/${merge_jobname}_%J.err

source ~/miniforge3/etc/profile.d/conda.sh
conda activate freqrise

lscpu
echo "-----------------------------------------"
python merge_outputs_from_jobarray.py \\
    --job_name $jobname \\
    --job_id $LSB_JOBID \\
    --output_path $output_path \\
    --dataset $dataset \\
    --noise_level $noise_level \\
    --synth_sig_len $synth_sig_len \\
    --labeltype $labeltype \\
    --n_samples $n_samples \\
EOF

# Submit the job
bsub < merge_outputs.sh

# Clean up
rm merge_outputs.sh