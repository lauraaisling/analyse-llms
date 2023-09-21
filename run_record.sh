#!/bin/bash

# bash run_record.sh
source ~/anaconda3/bin/activate analyse-llms1

# python scripts/compute_data.py --model_config_path preset_configs/pythia70m-base.json --data_path data/val.jsonl.zst --calculation_output_path outputs/calculation_data_probs_50000-pythia70m-base.p
# python scripts/compute_data.py --model_config_path preset_configs/pythia70m-sft.json --data_path data/val.jsonl.zst --calculation_output_path outputs/calculation_data_probs_50000-pythia70m-sft.p
# python scripts/compute_data.py --model_config_path preset_configs/pythia70m-dpo.json --data_path data/val.jsonl.zst --calculation_output_path outputs/calculation_data_probs_50000-pythia70m-dpo.p

# python scripts/calc_entropy_perplexity.py --calculation_output_path outputs/calculation_data_probs_50000-pythia70m-base.p --output_path outputs/perplexity_50000-pythia70m-base.json
# python scripts/calc_entropy_perplexity.py --calculation_output_path outputs/calculation_data_probs_50000_pythia70m-sft.p --output_path outputs/perplexity_50000-pythia70m-sft.json
# python scripts/calc_entropy_perplexity.py --calculation_output_path outputs/calculation_data_probs_50000_pythia70m-dpo.p --output_path outputs/perplexity_50000-pythia70m-dpo.json


# python scripts/compute_data.py --model_config_path preset_configs/pythia160m-base.json --data_path data/val.jsonl.zst --calculation_output_path outputs/calculation_data_all_20000-pythia160m-base.p --calc_probs True --calc_confidence True --max_docs 20000
# python scripts/compute_data.py --model_config_path preset_configs/pythia160m-sft.json --data_path data/val.jsonl.zst --calculation_output_path outputs/calculation_data_all_20000-pythia160m-sft.p --calc_probs True --calc_confidence True --max_docs 20000
# python scripts/compute_data.py --model_config_path preset_configs/pythia160m-dpo.json --data_path data/val.jsonl.zst --calculation_output_path outputs/calculation_data_all_20000-pythia160m-dpo.p --calc_probs True --calc_confidence True --max_docs 20000


# python scripts/calc_entropy_perplexity.py --calculation_output_path outputs/calculation_data_all_20000-pythia160m-base.p --output_path outputs/perplexity_20000-pythia160m-base.json
# python scripts/calc_entropy_perplexity.py --calculation_output_path outputs/calculation_data_all_20000-pythia160m-sft.p --output_path outputs/perplexity_20000-pythia160m-sft.json
# python scripts/calc_entropy_perplexity.py --calculation_output_path outputs/calculation_data_all_20000-pythia160m-dpo.p --output_path outputs/perplexity_20000-pythia160m-dpo.json


# python scripts/compute_data.py --model_config_path preset_configs/pythia410m-base.json --data_path data/val.jsonl.zst --calculation_output_path outputs/calculation_data_all_20000-pythia410m-base.p --calc_probs True --calc_confidence True --max_docs 20000
# python scripts/compute_data.py --model_config_path preset_configs/pythia410m-sft.json --data_path data/val.jsonl.zst --calculation_output_path outputs/calculation_data_all_20000-pythia410m-sft.p --calc_probs True --calc_confidence True --max_docs 20000
# python scripts/compute_data.py --model_config_path preset_configs/pythia410m-dpo.json --data_path data/val.jsonl.zst --calculation_output_path outputs/calculation_data_all_20000-pythia410m-dpo.p --calc_probs True --calc_confidence True --max_docs 20000


# python scripts/calc_entropy_perplexity.py --calculation_output_path outputs/calculation_data_all_20000-pythia410m-base.p --output_path outputs/perplexity_20000-pythia410m-base.json
# python scripts/calc_entropy_perplexity.py --calculation_output_path outputs/calculation_data_all_20000-pythia410m-sft.p --output_path outputs/perplexity_20000-pythia410m-sft.json
# python scripts/calc_entropy_perplexity.py --calculation_output_path outputs/calculation_data_all_20000-pythia410m-dpo.p --output_path outputs/perplexity_20000-pythia410m-dpo.json

# python scripts/compute_data.py --model_config_path preset_configs/pythia410m-base.json --data_path data/val.jsonl.zst --calculation_output_path outputs/calculation_data_ent_20000-pythia410m-base.p --calc_probs True --calc_confidence True --max_docs 20000
# python scripts/compute_data.py --model_config_path preset_configs/pythia410m-sft.json --data_path data/val.jsonl.zst --calculation_output_path outputs/calculation_data_ent_20000-pythia410m-sft.p --calc_probs True --calc_confidence True --max_docs 20000
# python scripts/compute_data.py --model_config_path preset_configs/pythia410m-dpo.json --data_path data/val.jsonl.zst --calculation_output_path outputs/calculation_data_ent_20000-pythia410m-dpo.p --calc_probs True --calc_confidence True --max_docs 20000

# python scripts/calc_entropy_perplexity.py --calculation_output_path outputs/calculation_data_ent_20000-pythia410m-base.p --output_path outputs/perplexity_20000-pythia410m-base.json
# python scripts/calc_entropy_perplexity.py --calculation_output_path outputs/calculation_data_ent_20000-pythia410m-sft.p --output_path outputs/perplexity_20000-pythia410m-sft.json
# python scripts/calc_entropy_perplexity.py --calculation_output_path outputs/calculation_data_ent_20000-pythia410m-dpo.p --output_path outputs/perplexity_20000-pythia410m-dpo.json


# bash scripts/kaggle_api.sh
# bash scripts/download_jigsaw_toxic.sh


# python scripts/jigsaw_toxic_hs.py --hf_model EleutherAI/pythia-2.8b --n 1000 --layers=[-1,-2,-3,-4,-5]
# python scripts/jigsaw_toxic_hs.py --hf_model lomahony/eleuther-pythia2.8b-hh-sft --n 1000 --layers=[-1,-2,-3,-4,-5]
# python scripts/jigsaw_toxic_hs.py --hf_model lomahony/eleuther-pythia2.8b-hh-dpo --n 1000 --layers=[-1,-2,-3,-4,-5]


# python scripts/jigsaw_toxic_hs.py --hf_model EleutherAI/pythia-2.8b --n 1000 --layers=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
# python scripts/jigsaw_toxic_hs.py --hf_model lomahony/eleuther-pythia2.8b-hh-sft --n 1000 --layers=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
# python scripts/jigsaw_toxic_hs.py --hf_model lomahony/eleuther-pythia2.8b-hh-dpo --n 1000 --layers=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]

# bash run_record.sh 