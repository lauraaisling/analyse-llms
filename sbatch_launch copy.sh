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

### RUNNING 

# echo 'perplexity calcs 28b-dpo not bf16'
# python scripts/compute_data.py --model_config_path preset_configs/pythia28b-dpo.json --data_path data/val.jsonl --calculation_output_path outputs/pythia28b-dpo-calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000
# python scripts/calc_entropy_perplexity.py --calculation_output_path outputs/pythia28b-dpo-calculation_data10000pc.p --output_path results/pythia28b-dpo-perplexity.json

# echo 'creative_and_factual_metrics 5 words 6.9b-sft'
# python scripts/creative_and_factual_metrics.py --model "pythia-6.9b-sft" --factual_completion_path "results/pythia-6.9b-sft_completions_factual_max_length70.npy" --creative_completion_path "results/pythia-6.9b-sft_completions_creative_max_length70.npy" --max_num_words 5
# echo 'creative_and_factual_metrics 5 words 6.9b-dpo'
# python scripts/creative_and_factual_metrics.py --model "pythia-6.9b-dpo" --factual_completion_path "results/pythia-6.9b-dpo_completions_factual_max_length70.npy" --creative_completion_path "results/pythia-6.9b-dpo_completions_creative_max_length70.npy" --max_num_words 5

# echo 'creative_and_factual_metrics 20 words 6.9b-sft'
# python scripts/creative_and_factual_metrics.py --model "pythia-6.9b-sft" --factual_completion_path "results/pythia-6.9b-sft_completions_factual_max_length70.npy" --creative_completion_path "results/pythia-6.9b-sft_completions_creative_max_length70.npy"
# echo 'creative_and_factual_metrics 20 words 6.9b-dpo'
# python scripts/creative_and_factual_metrics.py --model "pythia-6.9b-dpo" --factual_completion_path "results/pythia-6.9b-dpo_completions_factual_max_length70.npy" --creative_completion_path "results/pythia-6.9b-dpo_completions_creative_max_length70.npy"
# echo 'creative_and_factual_metrics 20 words 6.9b-ppo'
# python scripts/creative_and_factual_metrics.py --model "pythia-6.9b-ppo" --factual_completion_path "results/pythia-6.9b-ppo_completions_factual_max_length70.npy" --creative_completion_path "results/pythia-6.9b-ppo_completions_creative_max_length70.npy"

### TO RUN
# echo 'save_results 2.8b'
# python scripts/save_results.py --data_path "outputs/pythia28b-base-calculation_data10000pc.p,outputs/pythia28b-sft-calculation_data10000pc.p,outputs/pythia28b-dpo-calculation_data10000pc.p" --labels "PLM,SFT,DPO" --txt_label "2.8b"
