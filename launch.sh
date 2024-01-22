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

# 1b no probs or confidence
python scripts/compute_data.py --model_config_path preset_configs/pythia1b-base.json --data_path data/val.jsonl --calculation_output_path outputs/pythia1b-base-calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000
python scripts/compute_data.py --model_config_path preset_configs/pythia1b-sft.json --data_path data/val.jsonl --calculation_output_path outputs/pythia1b-sft-calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000
python scripts/compute_data.py --model_config_path preset_configs/pythia1b-dpo.json --data_path data/val.jsonl --calculation_output_path outputs/pythia1b-dpo-calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000

# perplexity 1b
python scripts/calc_entropy_perplexity.py --calculation_output_path outputs/pythia1b-base-calculation_data10000pc.p --output_path results/pythia1b-base-perplexity.json
python scripts/calc_entropy_perplexity.py --calculation_output_path outputs/pythia1b-sft-calculation_data10000pc.p --output_path results/pythia1b-sft-perplexity.json
python scripts/calc_entropy_perplexity.py --calculation_output_path outputs/pythia1b-dpo-calculation_data10000pc.p --output_path results/pythia1b-dpo-perplexity.json

# 1.4b no probs or confidence
# python scripts/compute_data.py --model_config_path preset_configs/pythia14b-base.json --data_path data/val.jsonl --calculation_output_path outputs/pythia14b-base-calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000
# python scripts/compute_data.py --model_config_path preset_configs/pythia14b-sft.json --data_path data/val.jsonl --calculation_output_path outputs/pythia14b-sft-calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000
# python scripts/compute_data.py --model_config_path preset_configs/pythia14b-dpo.json --data_path data/val.jsonl --calculation_output_path outputs/pythia14b-dpo-calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000

# # perplexity 1.4b
# python scripts/calc_entropy_perplexity.py --calculation_output_path outputs/pythia14b-base-calculation_data10000pc.p --output_path results/pythia14b-base-perplexity.json
# python scripts/calc_entropy_perplexity.py --calculation_output_path outputs/pythia14b-sft-calculation_data10000pc.p --output_path results/pythia14b-sft-perplexity.json
# python scripts/calc_entropy_perplexity.py --calculation_output_path outputs/pythia14b-dpo-calculation_data10000pc.p --output_path results/pythia14b-dpo-perplexity.json

# 2.8b no probs or confidence
# python scripts/compute_data.py --model_config_path preset_configs/pythia28b-base.json --data_path data/val.jsonl --calculation_output_path outputs/pythia28b-base-calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000
# python scripts/compute_data.py --model_config_path preset_configs/pythia28b-sft.json --data_path data/val.jsonl --calculation_output_path outputs/pythia28b-sft-calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000
# python scripts/compute_data.py --model_config_path preset_configs/pythia28b-dpo.json --data_path data/val.jsonl --calculation_output_path outputs/pythia28b-dpo-calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000

# perplexity 2.8b
# python scripts/calc_entropy_perplexity.py --calculation_output_path outputs/pythia28b-base-calculation_data10000pc.p --output_path results/pythia28b-base-perplexity.json
# python scripts/calc_entropy_perplexity.py --calculation_output_path outputs/pythia28b-sft-calculation_data10000pc.p --output_path results/pythia28b-sft-perplexity.json
# python scripts/calc_entropy_perplexity.py --calculation_output_path outputs/pythia28b-dpo-calculation_data10000pc.p --output_path results/pythia28b-dpo-perplexity.json

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

# python -c """print("llama-7b")"""
# python scripts/compute_data.py --model_config_path preset_configs/llama-7b.json --data_path data/val.jsonl --calculation_output_path outputs/llama-7b-base_calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000
# python -c """print("llama-7b-chat")""
# python scripts/compute_data.py --model_config_path preset_configs/llama-7b-chat.json --data_path data/val.jsonl --calculation_output_path outputs/llama-7b-chat_calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000

# python scripts/calc_entropy_perplexity.py --calculation_output_path outputs/llama-7b-base_calculation_data10000pc.p --output_path results/llama-7b-perplexity.json
# python scripts/calc_entropy_perplexity.py --calculation_output_path outputs/llama-7b-chat_calculation_data10000pc.p --output_path results/llama-7b-chat-perplexity.json
