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

### 
# 1b no probs or confidence
# python scripts/compute_data.py --model_config_path preset_configs/pythia1b-base.json --data_path data/val.jsonl --calculation_output_path outputs/pythia1b-base-calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000
# python scripts/compute_data.py --model_config_path preset_configs/pythia1b-sft.json --data_path data/val.jsonl --calculation_output_path outputs/pythia1b-sft-calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000
# python scripts/compute_data.py --model_config_path preset_configs/pythia1b-dpo.json --data_path data/val.jsonl --calculation_output_path outputs/pythia1b-dpo-calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000
# # perplexity 1b
# python scripts/calc_entropy_perplexity.py --calculation_output_path outputs/pythia1b-base-calculation_data10000pc.p --output_path results/pythia1b-base-perplexity.json
# python scripts/calc_entropy_perplexity.py --calculation_output_path outputs/pythia1b-sft-calculation_data10000pc.p --output_path results/pythia1b-sft-perplexity.json
# python scripts/calc_entropy_perplexity.py --calculation_output_path outputs/pythia1b-dpo-calculation_data10000pc.p --output_path results/pythia1b-dpo-perplexity.json

### 
# 1.4b no probs or confidence
# python scripts/compute_data.py --model_config_path preset_configs/pythia14b-base.json --data_path data/val.jsonl --calculation_output_path outputs/pythia14b-base-calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000
# python scripts/compute_data.py --model_config_path preset_configs/pythia14b-sft.json --data_path data/val.jsonl --calculation_output_path outputs/pythia14b-sft-calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000
# python scripts/compute_data.py --model_config_path preset_configs/pythia14b-dpo.json --data_path data/val.jsonl --calculation_output_path outputs/pythia14b-dpo-calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000
# # perplexity 1.4b
# python scripts/calc_entropy_perplexity.py --calculation_output_path outputs/pythia14b-base-calculation_data10000pc.p --output_path results/pythia14b-base-perplexity.json
# python scripts/calc_entropy_perplexity.py --calculation_output_path outputs/pythia14b-sft-calculation_data10000pc.p --output_path results/pythia14b-sft-perplexity.json
# python scripts/calc_entropy_perplexity.py --calculation_output_path outputs/pythia14b-dpo-calculation_data10000pc.p --output_path results/pythia14b-dpo-perplexity.json

### 
# 2.8b no probs or confidence
# python scripts/compute_data.py --model_config_path preset_configs/pythia28b-base.json --data_path data/val.jsonl --calculation_output_path outputs/pythia28b-base-calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000
# python scripts/compute_data.py --model_config_path preset_configs/pythia28b-sft.json --data_path data/val.jsonl --calculation_output_path outputs/pythia28b-sft-calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000
###
# python -c """print("28b-dpo")"""
# python scripts/compute_data.py --model_config_path preset_configs/pythia28b-dpo.json --data_path data/val.jsonl --calculation_output_path outputs/pythia28b-dpo-calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000
# # perplexity 2.8b
# python scripts/calc_entropy_perplexity.py --calculation_output_path outputs/pythia28b-base-calculation_data10000pc.p --output_path results/pythia28b-base-perplexity.json
# python scripts/calc_entropy_perplexity.py --calculation_output_path outputs/pythia28b-sft-calculation_data10000pc.p --output_path results/pythia28b-sft-perplexity.json
# python scripts/calc_entropy_perplexity.py --calculation_output_path outputs/pythia28b-dpo-calculation_data10000pc.p --output_path results/pythia28b-dpo-perplexity.json

### 
# 6.9b no probs or confidence
# python scripts/compute_data.py --model_config_path preset_configs/pythia69b-base.json --data_path data/val.jsonl --calculation_output_path outputs/pythia69b-base-calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000
# python scripts/compute_data.py --model_config_path preset_configs/pythia69b-sft.json --data_path data/val.jsonl --calculation_output_path outputs/pythia69b-sft-calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000
###
# python -c """print("69b-dpo")"""
# python scripts/compute_data.py --model_config_path preset_configs/pythia69b-dpo.json --data_path data/val.jsonl --calculation_output_path outputs/pythia69b-dpo-calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000
###
# python -c """print("69b-ppo")"""
# python scripts/compute_data.py --model_config_path preset_configs/pythia69b-ppo.json --data_path data/val.jsonl --calculation_output_path outputs/pythia69b-ppo-calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000
# # perplexity 6.9b
# python scripts/calc_entropy_perplexity.py --calculation_output_path outputs/pythia69b-base-calculation_data10000pc.p --output_path results/pythia69b-base-perplexity.json
# python scripts/calc_entropy_perplexity.py --calculation_output_path outputs/pythia69b-sft-calculation_data10000pc.p --output_path results/pythia69b-sft-perplexity.json
# python scripts/calc_entropy_perplexity.py --calculation_output_path outputs/pythia69b-dpo-calculation_data10000pc.p --output_path results/pythia69b-dpo-perplexity.json
# python scripts/calc_entropy_perplexity.py --calculation_output_path outputs/pythia69b-ppo-calculation_data10000pc.p --output_path results/pythia69b-ppo-perplexity.json

# python scripts/compute_data.py --model_config_path preset_configs/llama-7b.json --data_path data/val.jsonl --calculation_output_path outputs/llama-7b-base_calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000
# python scripts/compute_data.py --model_config_path preset_configs/llama-7b-chat.json --data_path data/val.jsonl --calculation_output_path outputs/llama-7b-chat_calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000
# # perplexity
# python scripts/calc_entropy_perplexity.py --calculation_output_path outputs/llama-7b-base_calculation_data10000pc.p --output_path results/llama-7b-perplexity.json
# python scripts/calc_entropy_perplexity.py --calculation_output_path outputs/llama-7b-chat_calculation_data10000pc.p --output_path results/llama-7b-chat-perplexity.json

# python scripts/creative_and_factual_completions.py
# python scripts/creative_and_factual_metrics.py --model "llama2-chat" --factual_completion_path "results/llama2-chat_completions_factual_max_length70.npy" --creative_completion_path "results/llama2-chat_completions_creative_max_length70.npy"
# python scripts/creative_and_factual_metrics.py --model "llama2" --factual_completion_path "results/llama2_completions_factual_max_length70.npy" --creative_completion_path "results/llama2_completions_creative_max_length70.npy"
###
# python -c """print("creative_and_factual_metrics 20 words 6.9b")"""
# python scripts/creative_and_factual_metrics.py --model "pythia-6.9b" --factual_completion_path "results/pythia-6.9b_completions_factual_max_length70.npy" --creative_completion_path "results/pythia-6.9b_completions_creative_max_length70.npy"
# python scripts/creative_and_factual_metrics.py --model "pythia-6.9b-sft" --factual_completion_path "results/pythia-6.9b-sft_completions_factual_max_length70.npy" --creative_completion_path "results/pythia-6.9b-sft_completions_creative_max_length70.npy"
# python scripts/creative_and_factual_metrics.py --model "pythia-6.9b-dpo" --factual_completion_path "results/pythia-6.9b-dpo_completions_factual_max_length70.npy" --creative_completion_path "results/pythia-6.9b-dpo_completions_creative_max_length70.npy"
# python scripts/creative_and_factual_metrics.py --model "pythia-6.9b-ppo" --factual_completion_path "results/pythia-6.9b-ppo_completions_factual_max_length70.npy" --creative_completion_path "results/pythia-6.9b-ppo_completions_creative_max_length70.npy"

###
# python -c """print("creative_and_factual_metrics 5 words llama, 6.9b")"""
# python scripts/creative_and_factual_metrics.py --model "llama2-chat" --factual_completion_path "results/llama2-chat_completions_factual_max_length70.npy" --creative_completion_path "results/llama2-chat_completions_creative_max_length70.npy" --max_num_words 5
# python scripts/creative_and_factual_metrics.py --model "llama2" --factual_completion_path "results/llama2_completions_factual_max_length70.npy" --creative_completion_path "results/llama2_completions_creative_max_length70.npy" --max_num_words 5
# python scripts/creative_and_factual_metrics.py --model "pythia-6.9b" --factual_completion_path "results/pythia-6.9b_completions_factual_max_length70.npy" --creative_completion_path "results/pythia-6.9b_completions_creative_max_length70.npy" --max_num_words 5
# python scripts/creative_and_factual_metrics.py --model "pythia-6.9b-sft" --factual_completion_path "results/pythia-6.9b-sft_completions_factual_max_length70.npy" --creative_completion_path "results/pythia-6.9b-sft_completions_creative_max_length70.npy" --max_num_words 5
# python scripts/creative_and_factual_metrics.py --model "pythia-6.9b-dpo" --factual_completion_path "results/pythia-6.9b-dpo_completions_factual_max_length70.npy" --creative_completion_path "results/pythia-6.9b-dpo_completions_creative_max_length70.npy" --max_num_words 5
# python scripts/creative_and_factual_metrics.py --model "pythia-6.9b-ppo" --factual_completion_path "results/pythia-6.9b-ppo_completions_factual_max_length70.npy" --creative_completion_path "results/pythia-6.9b-ppo_completions_creative_max_length70.npy" --max_num_words 5


# python scripts/save_results.py --data_path "outputs/pythia1b-base-calculation_data10000pc.p,outputs/pythia1b-sft-calculation_data10000pc.p,outputs/pythia1b-dpo-calculation_data10000pc.p" --labels "PLM,SFT,DPO" --txt_label "1b"
# python scripts/save_results.py --data_path "outputs/pythia14b-base-calculation_data10000pc.p,outputs/pythia14b-sft-calculation_data10000pc.p,outputs/pythia14b-dpo-calculation_data10000pc.p" --labels "PLM,SFT,DPO" --txt_label "1.4b"
### python -c """print("save_results 2.8b, 6.9b")"""
### python scripts/save_results.py --data_path "outputs/pythia28b-base-calculation_data10000pc.p,outputs/pythia28b-sft-calculation_data10000pc.p,outputs/pythia28b-dpo-calculation_data10000pc.p" --labels "PLM,SFT,DPO" --txt_label "2.8b"
### python scripts/save_results.py --data_path "outputs/pythia69b-base-calculation_data10000pc.p,outputs/pythia69b-sft-calculation_data10000pc.p,outputs/pythia69b-dpo-calculation_data10000pc.p,outputs/pythia69b-ppo-calculation_data10000pc.p" --labels "PLM,SFT,DPO,PPO" --txt_label "6.9b"
