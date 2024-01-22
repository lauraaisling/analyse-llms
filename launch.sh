#!/bin/bash
# bash launch.sh

source ~/anaconda3/bin/activate analyse-llms

# CUDA_VISIBLE_DEVICES='2,3,4,5,6,7'

# 70m with probs and confidence
# python scripts/compute_data.py --model_config_path preset_configs/pythia70m-base.json --data_path data/val.jsonl --calculation_output_path outputs/pythia70m-base-calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000
# python scripts/compute_data.py --model_config_path preset_configs/pythia70m-sft.json --data_path data/val.jsonl --calculation_output_path outputs/pythia70m-sft-calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000
# python scripts/compute_data.py --model_config_path preset_configs/pythia70m-dpo.json --data_path data/val.jsonl --calculation_output_path outputs/pythia70m-dpo-calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000

# 410m with probs and confidence
# python scripts/compute_data.py --model_config_path preset_configs/pythia410m-base.json --data_path data/val.jsonl --calculation_output_path outputs/pythia410m-base-calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000
# python scripts/compute_data.py --model_config_path preset_configs/pythia410m-sft.json --data_path data/val.jsonl --calculation_output_path outputs/pythia410m-sft-calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000
# python scripts/compute_data.py --model_config_path preset_configs/pythia410m-dpo.json --data_path data/val.jsonl --calculation_output_path outputs/pythia410m-dpo-calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000

# perplexity 70m
# python scripts/calc_entropy_perplexity.py --calculation_output_path outputs/pythia70m-base-calculation_data10000pc.p --output_path results/pythia70m-base-perplexity.json
# python scripts/calc_entropy_perplexity.py --calculation_output_path outputs/pythia70m-sft-calculation_data10000pc.p --output_path results/pythia70m-sft-perplexity.json
# python scripts/calc_entropy_perplexity.py --calculation_output_path outputs/pythia70m-dpo-calculation_data10000pc.p --output_path results/pythia70m-dpo-perplexity.json

# perplexity 410m
# python scripts/calc_entropy_perplexity.py --calculation_output_path outputs/pythia410m-base-calculation_data10000pc.p --output_path results/pythia410m-base-perplexity.json
# python scripts/calc_entropy_perplexity.py --calculation_output_path outputs/pythia410m-sft-calculation_data10000pc.p --output_path results/pythia410m-sft-perplexity.json
# python scripts/calc_entropy_perplexity.py --calculation_output_path outputs/pythia410m-dpo-calculation_data10000pc.p --output_path results/pythia410m-dpo-perplexity.json

# 160m no probs or confidence
# python scripts/compute_data.py --model_config_path preset_configs/pythia160m-base.json --data_path data/val.jsonl --calculation_output_path outputs/pythia160m-base-calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000
# python scripts/compute_data.py --model_config_path preset_configs/pythia160m-sft.json --data_path data/val.jsonl --calculation_output_path outputs/pythia160m-sft-calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000
# python scripts/compute_data.py --model_config_path preset_configs/pythia160m-dpo.json --data_path data/val.jsonl --calculation_output_path outputs/pythia160m-dpo-calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000

# perplexity 160m
# python scripts/calc_entropy_perplexity.py --calculation_output_path outputs/pythia160m-base-calculation_data10000pc.p --output_path results/pythia160m-base-perplexity.json
# python scripts/calc_entropy_perplexity.py --calculation_output_path outputs/pythia160m-sft-calculation_data10000pc.p --output_path results/pythia160m-sft-perplexity.json
# python scripts/calc_entropy_perplexity.py --calculation_output_path outputs/pythia160m-dpo-calculation_data10000pc.p --output_path results/pythia160m-dpo-perplexity.json

