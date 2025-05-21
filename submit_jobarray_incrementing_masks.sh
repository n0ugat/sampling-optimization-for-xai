#!/bin/bash

# Define the variables
# ! Manually add the tag "--incrementing_masks" to both the jobarray and merge job if you want to use the incrementing masks
# ! The same applies to the tag "--no_random_peaks" if you want no random peaks in the synthetic data
jobname="incrementing_masks_Synthetic"
merge_jobname="merge_${jobname}_Ssm_m"
evaluation_jobname="${jobname}_Ssm_e"
plotting_jobname="${jobname}_Ssm_p"

dataset="synthetic"
output_path="outputs"
n_samples=10

# If using AudioMNIST
labeltype="gender" 

# If using synthetic
noise_level=0.0
synth_sig_len=100

# Generate the jobarray.sh
cat <<EOF > jobarray.sh
#!/bin/bash
#BSUB -q hpc
#BSUB -J ${jobname}[1-13]
#BSUB -n 4
#BSUB -W 06:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=1GB]"
#BSUB -o outputs/hpclogs/jobarrays/${jobname}_%J_%I.out
#BSUB -e outputs/hpclogs/jobarrays/${jobname}_%J_%I.err

source ~/miniforge3/etc/profile.d/conda.sh
conda activate freqrise

lscpu
echo "-----------------------------------------"
echo "Job idx: \$LSB_JOBINDEX, Job ID: \${LSB_JOBID}, Job name: $jobname"
echo "-----------------------------------------"
python main_attributions.py \\
    --job_idx \$LSB_JOBINDEX \\
    --job_name $jobname \\
    --data_path data/ \\
    --model_path models \\
    --output_path $output_path \\
    --dataset $dataset \\
    --noise_level $noise_level \\
    --synth_sig_len $synth_sig_len \\
    --labeltype $labeltype \\
    --n_samples $n_samples \\
    --incrementing_masks \\
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
#BSUB -W 00:30
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
    --job_id \$LSB_JOBID \\
    --output_path $output_path \\
    --dataset $dataset \\
    --noise_level $noise_level \\
    --synth_sig_len $synth_sig_len \\
    --labeltype $labeltype \\
    --n_samples $n_samples \\
    --incrementing_masks \\
EOF

# Submit the job
bsub < merge_outputs.sh

# Clean up
rm merge_outputs.sh


# Evaluate the method
cat <<EOF > evaluation_sm.sh
#!/bin/bash
#BSUB -q hpc
#BSUB -J ${evaluation_jobname}
#BSUB -n 1
#BSUB -W 01:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -w ended(${merge_jobname})
#BSUB -o outputs/hpclogs/${evaluation_jobname}_%J.out
#BSUB -e outputs/hpclogs/${evaluation_jobname}_%J.err

source ~/miniforge3/etc/profile.d/conda.sh
conda activate freqrise

lscpu
echo "-----------------------------------------"
python main_evaluation.py \\
    --output_path $output_path \\
    --dataset $dataset \\
    --noise_level $noise_level \\
    --synth_sig_len $synth_sig_len \\
    --labeltype $labeltype \\
    --n_samples $n_samples \\
    --incrementing_masks \\
    --compute_deletion_scores \\
    --compute_complexity_scores \\
    --compute_localization_scores \\
EOF

# Submit the job
bsub < evaluation_sm.sh

# Clean up
rm evaluation_sm.sh



# Plot the scores for the incrementing masks
cat <<EOF > plotting.sh
#!/bin/bash
#BSUB -q hpc
#BSUB -J ${plotting_jobname}
#BSUB -n 1
#BSUB -W 00:10
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2GB]"
#BSUB -w ended(${evaluation_jobname})
#BSUB -o outputs/hpclogs/${plotting_jobname}_%J.out
#BSUB -e outputs/hpclogs/${plotting_jobname}_%J.err

source ~/miniforge3/etc/profile.d/conda.sh
conda activate freqrise

lscpu
echo "-----------------------------------------"
python src/plotting/increment_masks_plotting.py \\
    --output_path $output_path \\
    --dataset $dataset \\
    --noise_level $noise_level \\
    --synth_sig_len $synth_sig_len \\
    --labeltype $labeltype \\
    --n_samples $n_samples \\
EOF

# Submit the job
bsub < plotting.sh

# Clean up
rm plotting.sh