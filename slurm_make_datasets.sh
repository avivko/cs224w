#!/bin/bash
#SBATCH --job-name=mk_datasets
#SBATCH --partition=owners
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=16GB
#SBATCH --error=/home/users/kormanav/slurm_jobs/job_%J.err
#SBATCH --output=/home/users/kormanav/slurm_jobs/job_%J.out

module load cuda/11.3.1

source $HOME/.bashrc
conda activate cs224wproj

working_dir=$HOME/cs224w
export TORCH_HOME=$working_dir/torch_home

#identifier=known_simple_Aug3
#model=GCN
#epochs=50

python working_dir/cs224w/src/multi_configs_to_pyg.py