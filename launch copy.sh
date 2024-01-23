#!/bin/bash
# bash launch.sh

source ~/anaconda3/bin/activate analyse-llms

# CUDA_VISIBLE_DEVICES='2,3,4,5,6,7'


# echo '28b-dpo'
# python scripts/compute_data.py --model_config_path preset_configs/pythia28b-dpo.json --data_path data/val.jsonl --calculation_output_path outputs/pythia28b-dpo-calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000
# python scripts/calc_entropy_perplexity.py --calculation_output_path outputs/pythia28b-dpo-calculation_data10000pc.p --output_path results/pythia28b-dpo-perplexity.json

# python scripts/save_results.py --data_path "outputs/pythia70m-base-calculation_data10000pc.p,outputs/pythia70m-sft-calculation_data10000pc.p,outputs/pythia70m-dpo-calculation_data10000pc.p" --labels "PLM,SFT,DPO" --txt_label "70m"
# python scripts/save_results.py --data_path "outputs/pythia160m-base-calculation_data10000pc.p,outputs/pythia160m-sft-calculation_data10000pc.p,outputs/pythia160m-dpo-calculation_data10000pc.p" --labels "PLM,SFT,DPO" --txt_label "160m"
# python scripts/save_results.py --data_path "outputs/pythia410m-base-calculation_data10000pc.p,outputs/pythia410m-sft-calculation_data10000pc.p,outputs/pythia410m-dpo-calculation_data10000pc.p" --labels "PLM,SFT,DPO" --txt_label "410m"
