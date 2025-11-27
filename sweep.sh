#!/bin/bash
#SBATCH -o /home/mila/m/marco.stock/slurm_logs/slurm-%j.out
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=1000G                     # server memory requested (per node)
#SBATCH -t 03:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=short-unkillable               # Request partition
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:l40s:4               # Type/number of GPUs needed
#SBATCH -c 6

# 1. Load the required modules
module --quiet load miniconda/3

# 2. Load your environment
conda activate ptgeo
git clone --depth 1 -b "master" https://github.com/ScialdoneLab/Reevaluating-GRN-inference.git $SLURM_TMPDIR/

# 3. Launch your job
cd $SLURM_TMPDIR/
cp -r $HOME/granpy/data .

srun wandb agent --count 200 scialdonelab/GRN_inference/pidocht1