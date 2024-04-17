#!/bin/bash

# Arrays of hyperparameters to grid search over
lrs=("3e-4")
weight_decays=("1e-2" "0")
loss_mode=("mean" "sum")
lora=("True")

# Nested loops to iterate over all combinations of hyperparameters
for lr in "${lrs[@]}"; do
  for weight_decay in "${weight_decays[@]}"; do
    for loss_mode in "${loss_mode[@]}"; do
      for lora in "${lora[@]}"; do


        # Generate a unique output file for each job to avoid overwriting
        output_file="/home/haolun/scratch/SNAKE/exp_out/E2_T5B_D2_full_lr${lr}_wd${weight_decay}_${loss_mode}_${lora}.out"

        # Launch a separate job for each hyperparameter combination
        sbatch <<EOL
#!/bin/bash
#SBATCH --account=def-cpsmcgil
#SBATCH --output=${output_file}
#SBATCH --gres=gpu:v100l:1
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000M

source /home/haolun/projects/def-cpsmcgil/SNAKE/venv_SNAKE/bin/activate
module load cuda
nvidia-smi

# Run your script with the current hyperparameter combination
WANDB__SERVICE_WAIT=300 python3 /home/haolun/scratch/SNAKE/experiment_E2.py \
  --model_choice="E2" \
  --dataset='gpt4' \
  --pretrained_model_name='t5-base' \
  --batch_size=2 \
  --epochs=100 \
  --log_wandb=True \
  --use_lora=${lora} \
  --lr=${lr} \
  --weight_decay=${weight_decay} \
  --mode='train' \
  --loss_mode=${loss_mode} \
  --use_better_init=True

deactivate
EOL

      done
    done
  done
done