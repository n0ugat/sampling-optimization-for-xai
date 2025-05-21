#!/bin/bash

jobname="Synthetic"

# Generate the jobscript.sh
cat <<EOF > jobscript.sh
#!/bin/bash
#BSUB -q gpuv100
#BSUB -J $jobname
#BSUB -n 8
#BSUB -W 20:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=${memory_per_core}MB]"
#BSUB -o outputs/hpclogs/${jobname}%J.out
#BSUB -e outputs/hpclogs/${jobname}%J.err

source ~/miniforge3/etc/profile.d/conda.sh
conda activate freqrise

lscpu
echo "-----------------------------------------"
python run_processes.py
EOF

# Submit the job
bsub < jobscript.sh

# Clean up
rm jobscript.sh