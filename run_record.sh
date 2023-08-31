#!/bin/bash

# bash run_record.sh
source ~/anaconda3/bin/activate analyse-llms

# python scripts/compute_data.py --model_config_path preset_configs/pythia70m-base.json --data_path data/val.jsonl.zst --calculation_output_path outputs/pythia70m-base_calculation_data_probs50000.p
# python scripts/compute_data.py --model_config_path preset_configs/pythia70m-sft.json --data_path data/val.jsonl.zst --calculation_output_path outputs/pythia70m-sft_calculation_data_probs50000.p
# python scripts/compute_data.py --model_config_path preset_configs/pythia70m-dpo.json --data_path data/val.jsonl.zst --calculation_output_path outputs/pythia70m-dpo_calculation_data_probs50000.p

# python scripts/calc_entropy_perplexity.py --calculation_output_path outputs/pythia70m-base_calculation_data_probs50000.p --output_path outputs/pythia70m-base-50000.json
# python scripts/calc_entropy_perplexity.py --calculation_output_path outputs/pythia70m-sft_calculation_data_probs50000.p --output_path outputs/pythia70m-sft-50000.json
# python scripts/calc_entropy_perplexity.py --calculation_output_path outputs/pythia70m-dpo_calculation_data_probs50000.p --output_path outputs/pythia70m-dpo-50000.json


python scripts/compute_data.py --model_config_path preset_configs/pythia160m-base.json --data_path data/val.jsonl.zst --calculation_output_path outputs/pythia160m-base_calculation_data_all_20000.p --calc_probs True --calc_confidence True --max_docs 20000
python scripts/compute_data.py --model_config_path preset_configs/pythia160m-sft.json --data_path data/val.jsonl.zst --calculation_output_path outputs/pythia160m-sft_calculation_data_all_20000.p --calc_probs True --calc_confidence True --max_docs 20000
python scripts/compute_data.py --model_config_path preset_configs/pythia160m-dpo.json --data_path data/val.jsonl.zst --calculation_output_path outputs/pythia160m-dpo_calculation_data_all_20000.p --calc_probs True --calc_confidence True --max_docs 20000
