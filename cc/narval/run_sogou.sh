#!/bin/bash
#SBATCH --account=ctb-lcharlin
#SBATCH --output="/home/haolun/projects/def-cpsmcgil/haolun/FairQoS/exp_out/sogou.out"
#SBATCH --gres=gpu:a100:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=40000M

source /home/haolun/projects/def-cpsmcgil/haolun/FairQoS/venv/bin/activate
module load cuda
nvidia-smi

# Run your script with the current hyperparameter combination
python3 /home/haolun/projects/def-cpsmcgil/haolun/FairQoS/run_full_process_pipeline.py --data_name=sogou --ncount=100

deactivate
EOL
