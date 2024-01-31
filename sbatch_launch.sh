#!/bin/bash
# sbatch sbatch_launch.sh
#Resource Request 
#SBATCH --account=eleuther
#SBATCH --job-name=modes
#SBATCH --output=slurm-TONAME-%x_%j.out   ## filename of the output; the %j is equivalent to jobID; default is slurm-[jobID].out 
#SBATCH --partition=a40x ## the partitions to run in (comma seperated) 
#SBATCH --gpus=1 # number of gpus per task 
#SBATCH --cpus-per-gpu=12 
#SBATCH --nodes=1
#SBATCH --mail-type=ALL

##SBATCH --error=%x_%jerror.out    # Set this dir where you want slurm outs to go
##SBATCH --ntasks=1  ## number of tasks (analyses) to run 
##SBATCH --gpus-per-task=1 # number of gpus per task
##SBATCH --ntasks-per-node=8

module load cuda/12.1

export HYDRA_FULL_ERROR=1

source ~/venvs/venv-analyse-llms/bin/activate

# ContextualAI/archangel_sft_llama7b
# ContextualAI/archangel_sft-ppo_llama7b
# ContextualAI/archangel_sft-dpo_llama7b

### Running
# echo 'completions llama-sft'
# python scripts/creative_and_factual_completions.py

echo 'diversity llama-sft 20 words'
python scripts/creative_and_factual_metrics.py --model "llama2-sft" --factual_completion_path "results/completions/llama2-sft_completions_factual_max_length70.npy" --creative_completion_path "results/completions/llama2-sft_completions_creative_max_length70.npy"

# echo 'diversity llama-sft 5 words'
# python scripts/creative_and_factual_metrics.py --model "llama2-sft" --factual_completion_path "results/completions/llama2-sft_completions_factual_max_length70.npy" --creative_completion_path "results/completions/llama2-sft_completions_creative_max_length70.npy" --max_num_words 5
