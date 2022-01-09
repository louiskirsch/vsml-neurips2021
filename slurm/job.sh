#!/bin/bash -l
#
#SBATCH --job-name=vsml
#SBATCH --time=04:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --signal=SIGINT@300

# Enable if compiled with CUDA compatible MPI
#export MPI4JAX_USE_CUDA_MPI=1

if [ -n "$WANDB_DEPENDENCY" ]
then
    export WANDB_RESUME='must'
    if [ -n "$SLURM_ARRAY_JOB_ID" ]
    then
        export WANDB_RUN_ID=`cat wandb-run-"$WANDB_DEPENDENCY"_"$SLURM_ARRAY_TASK_ID"`
    else
        export WANDB_RUN_ID=`cat wandb-run-"$WANDB_DEPENDENCY"`
    fi
fi


echo "Activate venv"
# TODO Set correct virtualenv path
source ~/path/to/venv/bin/activate

echo "Activate wandb"
# Make sure wandb is activated
sed -i '/^disabled = true/d' wandb/settings
sed -i '/^mode = offline/d' wandb/settings

echo "Run job"
if [[ "$@" == *"wandb agent"* ]]
then
  eval "$@"
else
  srun -X --wait=30 "$@"
fi
