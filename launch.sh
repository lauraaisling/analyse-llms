#!/bin/bash
# sbatch launch.sh
#Resource Request 
#SBATCH --account=eleuther
#SBATCH --job-name=modes
#SBATCH --output=slurm-TONAME-%x_%j.out   ## filename of the output; the %j is equivalent to jobID; default is slurm-[jobID].out 
#SBATCH --partition=g40x ## the partitions to run in (comma seperated) 

#SBATCH --gpus=1 # number of gpus per task 
#SBATCH --cpus-per-gpu=12 
#SBATCH --nodes=1

#SBATCH --mail-type=ALL

##SBATCH --error=%x_%jerror.out    # Set this dir where you want slurm outs to go
##SBATCH --ntasks=1  ## number of tasks (analyses) to run 
##SBATCH --gpus-per-task=1 # number of gpus per task 
##SBATCH --ntasks-per-node=8

module load cuda/11.7

export HYDRA_FULL_ERROR=1

source /fsx/home-laura/venv-analyse-llms/bin/activate

# python -c """print("pythia6.9b-base")"""
# python scripts/compute_data.py \
#     --model_config_path preset_configs/pythia6.9b-base.json \
#     --data_path /fsx/home-laura/pile-val/val.jsonl \
#     --calculation_output_path outputs/pythia6.9b-base_calculation_data.p \
#     --calc_confidence True \
#     --calc_probs True \
#     --max_docs 10000

# python -c """print("pythia2.8b-base")"""
# python scripts/compute_data.py \
#     --model_config_path preset_configs/pythia28-base.json \
#     --data_path /fsx/home-laura/pile-val/val.jsonl \
#     --calculation_output_path outputs/pythia2.8b-base_calculation_data.p \
#     --calc_confidence True \
#     --calc_probs True \
#     --max_docs 10000

# python -c """print("pythia2.8b-sft")"""
# python scripts/compute_data.py \
#     --model_config_path preset_configs/pythia28-sft.json \
#     --data_path /fsx/home-laura/pile-val/val.jsonl \
#     --calculation_output_path outputs/pythia2.8b-sft_calculation_data.p \
#     --calc_confidence True \
#     --calc_probs True \
#     --max_docs 10000

python -c """print("pythia160m-sft")"""
python scripts/compute_data.py \
    --model_config_path preset_configs/pythia160m-sft.json \
    --data_path /fsx/home-laura/pile-val/val.jsonl \
    --calculation_output_path outputs/pythia160m-sft_calculation_data.p \
    --calc_confidence True \
    --calc_probs True \
    --max_docs 10000

# python -c """print("llama-7b")"""
# python scripts/compute_data.py \
#     --model_config_path preset_configs/llama-7b.json \
#     --data_path /fsx/home-laura/pile-val/val.jsonl \
#     --calculation_output_path outputs/llama-7b_calculation_data.p \
#     --calc_confidence True \
#     --calc_probs True \
#     --max_docs 10000

# python -c """print("llama-7b-chat")"""
# python scripts/compute_data.py \
#     --model_config_path preset_configs/llama-7b-chat.json \
#     --data_path /fsx/home-laura/pile-val/val.jsonl \
#     --calculation_output_path outputs/llama-7b-chat-base_calculation_data.p \
#     --calc_confidence True \
#     --calc_probs True \
#     --max_docs 10000
