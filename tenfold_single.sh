#!/bin/bash
#SBATCH -o /home/mila/m/marco.stock/slurm_logs/slurm-%j.out
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=32G                     # server memory requested (per node)
#SBATCH -t 48:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=unkillable               # Request partition
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1               # Type/number of GPUs needed
#SBATCH -c 6

# 1. Load the required modules
module --quiet load miniconda/3

# 2. Load your environment
conda activate ptgeo
git clone --depth 1 -b "master" https://github.com/ScialdoneLab/Reevaluating-GRN-inference.git $SLURM_TMPDIR/

# 3. Launch your job
cd $SLURM_TMPDIR/
cp -r $HOME/granpy/data .

ID="$1"
if [ -z "$ID" ]; then
echo "Usage: $0 <id>"
exit 1
fi

srun --gres=gpu:1 python nfold.py --id "$ID" --missing_only