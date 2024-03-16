#!/bin/bash
# sbatch sbatch_launch.sh
#Resource Request 
#SBATCH --account=eleuther
#SBATCH --job-name=completions
#SBATCH --output=slurm_%x_pythia-2.8b-dpo_%j.out   ## filename of the output; the %j is equivalent to jobID; default is slurm-[jobID].out 
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
# source ~/venvs/analyse-llms/bin/activate

# ContextualAI/archangel_sft_llama7b
# ContextualAI/archangel_sft-ppo_llama7b
# ContextualAI/archangel_sft-dpo_llama7b

### Running
echo 'completions llama-slic'
python scripts/creative_and_factual_completions.py --model "llama2-slic"

# echo 'diversity llama-chat 20 words'
# python scripts/creative_and_factual_metrics.py --model "llama2-chat" --factual_completion_path "results/completions/llama2-chat_completions_factual_max_length25.npy" --creative_completion_path "results/completions/llama2-chat_completions_creative_max_length70.npy"
# echo 'diversity llama 20 words'
# python scripts/creative_and_factual_metrics.py --model "llama2" --factual_completion_path "results/completions/llama2_completions_factual_max_length25.npy" --creative_completion_path "results/completions/llama2_completions_creative_max_length70.npy"
# echo 'diversity llama-sft 20 words'
# python scripts/creative_and_factual_metrics.py --model "llama2-sft" --factual_completion_path "results/completions/llama2-sft_completions_factual_max_length25.npy" --creative_completion_path "results/completions/llama2-sft_completions_creative_max_length70.npy"
# echo 'diversity llama-dpo 20 words'
# python scripts/creative_and_factual_metrics.py --model "llama2-dpo" --factual_completion_path "results/completions/llama2-dpo_completions_factual_max_length25.npy" --creative_completion_path "results/completions/llama2-dpo_completions_creative_max_length70.npy"
# echo 'diversity llama-ppo 20 words'
# python scripts/creative_and_factual_metrics.py --model "llama2-ppo" --factual_completion_path "results/completions/llama2-ppo_completions_factual_max_length25.npy" --creative_completion_path "results/completions/llama2-ppo_completions_creative_max_length70.npy"
# echo 'diversity pythia-2.8b 20 words'
# python scripts/creative_and_factual_metrics.py --model "pythia-2.8b" --factual_completion_path "results/completions/pythia-2.8b_completions_factual_max_length25.npy" --creative_completion_path "results/completions/pythia-2.8b_completions_creative_max_length70.npy"
# echo 'diversity pythia-2.8b-sft 20 words'
# python scripts/creative_and_factual_metrics.py --model "pythia-2.8b-sft" --factual_completion_path "results/completions/pythia-2.8b-sft_completions_factual_max_length25.npy" --creative_completion_path "results/completions/pythia-2.8b-sft_completions_creative_max_length70.npy"
# echo 'diversity pythia-2.8b-dpo 20 words'
# python scripts/creative_and_factual_metrics.py --model "pythia-2.8b-dpo" --factual_completion_path "results/completions/pythia-2.8b-dpo_completions_factual_max_length25.npy" --creative_completion_path "results/completions/pythia-2.8b-dpo_completions_creative_max_length70.npy"

# echo 'diversity llama-kto 20 words'
# python scripts/creative_and_factual_metrics.py --model "llama2-kto" --factual_completion_path "results/completions/llama2-kto_completions_factual_max_length25.npy" --creative_completion_path "results/completions/llama2-kto_completions_creative_max_length70.npy"
echo 'diversity llama-slic 20 words'
python scripts/creative_and_factual_metrics.py --model "llama2-slic" --factual_completion_path "results/completions/llama2-slic_completions_factual_max_length25.npy" --creative_completion_path "results/completions/llama2-slic_completions_creative_max_length70.npy"

# echo 'diversity llama-sft 5 words'
# python scripts/creative_and_factual_metrics.py --model "llama2-kto" --factual_completion_path "results/completions/llama2-sft_completions_factual_max_length70.npy" --creative_completion_path "results/completions/llama2-sft_completions_creative_max_length70.npy" --max_num_words 5
