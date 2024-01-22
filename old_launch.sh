#!/bin/bash
# sbatch old_launch.sh
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
# python scripts/compute_data.py --model_config_path preset_configs/pythia28b-dpo.json --data_path data/val.jsonl --calculation_output_path outputs/pythia28b-dpo-calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000
# # perplexity 2.8b
# python scripts/calc_entropy_perplexity.py --calculation_output_path outputs/pythia28b-base-calculation_data10000pc.p --output_path results/pythia28b-base-perplexity.json
# python scripts/calc_entropy_perplexity.py --calculation_output_path outputs/pythia28b-sft-calculation_data10000pc.p --output_path results/pythia28b-sft-perplexity.json
# python scripts/calc_entropy_perplexity.py --calculation_output_path outputs/pythia28b-dpo-calculation_data10000pc.p --output_path results/pythia28b-dpo-perplexity.json

### 
# 6.9b no probs or confidence
# python scripts/compute_data.py --model_config_path preset_configs/pythia69b-base.json --data_path data/val.jsonl --calculation_output_path outputs/pythia69b-base-calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000
# python scripts/compute_data.py --model_config_path preset_configs/pythia69b-sft.json --data_path data/val.jsonl --calculation_output_path outputs/pythia69b-sft-calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000
# python scripts/compute_data.py --model_config_path preset_configs/pythia69b-dpo.json --data_path data/val.jsonl --calculation_output_path outputs/pythia69b-dpo-calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000
# python scripts/compute_data.py --model_config_path preset_configs/pythia69b-ppo.json --data_path data/val.jsonl --calculation_output_path outputs/pythia69b-ppo-calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000
# # perplexity 6.9b
# python scripts/calc_entropy_perplexity.py --calculation_output_path outputs/pythia69b-base-calculation_data10000pc.p --output_path results/pythia69b-base-perplexity.json
# python scripts/calc_entropy_perplexity.py --calculation_output_path outputs/pythia69b-sft-calculation_data10000pc.p --output_path results/pythia69b-sft-perplexity.json
# python scripts/calc_entropy_perplexity.py --calculation_output_path outputs/pythia69b-dpo-calculation_data10000pc.p --output_path results/pythia69b-dpo-perplexity.json
# python scripts/calc_entropy_perplexity.py --calculation_output_path outputs/pythia69b-ppo-calculation_data10000pc.p --output_path results/pythia69b-ppo-perplexity.json

### 
# python -c """print("llama-7b")"""
# python scripts/compute_data.py --model_config_path preset_configs/llama-7b.json --data_path data/val.jsonl --calculation_output_path outputs/llama-7b-base_calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000
# python -c """print("llama-7b-chat")"""
# python scripts/compute_data.py --model_config_path preset_configs/llama-7b-chat.json --data_path data/val.jsonl --calculation_output_path outputs/llama-7b-chat_calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000
# # perplexity
# python scripts/calc_entropy_perplexity.py --calculation_output_path outputs/llama-7b-base_calculation_data10000pc.p --output_path results/llama-7b-perplexity.json
# python scripts/calc_entropy_perplexity.py --calculation_output_path outputs/llama-7b-chat_calculation_data10000pc.p --output_path results/llama-7b-chat-perplexity.json

### 
# python scripts/creative_and_factual_calc.py

### 58024
python scripts/save_results.py --data_path "outputs/pythia1b-base-calculation_data10000pc.p,outputs/pythia14b-base-calculation_data10000pc.p" --labels "1b-PLM,1.4b-PLM" --txt_label "test"
