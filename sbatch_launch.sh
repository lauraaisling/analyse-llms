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

# echo 'creative_and_factual_completions 20 words 2.8b'
# python scripts/creative_and_factual_completions.py

### TO RUN
# echo 'creative_and_factual_metrics 20 words 6.9b-ppo'
# python scripts/creative_and_factual_metrics.py --model "pythia-6.9b-ppo" --factual_completion_path "results/pythia-6.9b-ppo_completions_factual_max_length70.npy" --creative_completion_path "results/pythia-6.9b-ppo_completions_creative_max_length70.npy"

# echo 'creative_and_factual_metrics 20 words 2.8b-plm'
# python scripts/creative_and_factual_metrics.py --model "pythia-2.8b" --factual_completion_path "results/pythia-2.8b_completions_factual_max_length70.npy" --creative_completion_path "results/pythia-2.8b_completions_creative_max_length70.npy"
# echo 'creative_and_factual_metrics 20 words 2.8b-sft'
# python scripts/creative_and_factual_metrics.py --model "pythia-2.8b-sft" --factual_completion_path "results/pythia-2.8b-sft_completions_factual_max_length70.npy" --creative_completion_path "results/pythia-2.8b-sft_completions_creative_max_length70.npy"
# echo 'creative_and_factual_metrics 20 words 2.8b-dpo'
# python scripts/creative_and_factual_metrics.py --model "pythia-2.8b-dpo" --factual_completion_path "results/pythia-2.8b-dpo_completions_factual_max_length70.npy" --creative_completion_path "results/pythia-2.8b-dpo_completions_creative_max_length70.npy"


# echo 'creative_and_factual_metrics 5 words 2.8b-plm'
# python scripts/creative_and_factual_metrics.py --model "pythia-2.8b" --factual_completion_path "results/pythia-2.8b_completions_factual_max_length70.npy" --creative_completion_path "results/pythia-2.8b_completions_creative_max_length70.npy" --max_num_words 5
# echo 'creative_and_factual_metrics 5 words 2.8b-sft'
# python scripts/creative_and_factual_metrics.py --model "pythia-2.8b-sft" --factual_completion_path "results/pythia-2.8b-sft_completions_factual_max_length70.npy" --creative_completion_path "results/pythia-2.8b-sft_completions_creative_max_length70.npy" --max_num_words 5
# echo 'creative_and_factual_metrics 5 words 2.8b-dpo'
# python scripts/creative_and_factual_metrics.py --model "pythia-2.8b-dpo" --factual_completion_path "results/pythia-2.8b-dpo_completions_factual_max_length70.npy" --creative_completion_path "results/pythia-2.8b-dpo_completions_creative_max_length70.npy" --max_num_words 5
