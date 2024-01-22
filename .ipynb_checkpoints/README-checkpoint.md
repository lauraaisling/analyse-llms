# analyse-llms

Exploratory analysis of the effects of fine tuning Pythia models. 

There are several notebooks to visualise script output as well as some random mode collapse experiments. 

## To compute token average probabilities (optional) and intermediate outputs for calculating entropy, perplexity (e.g. logprobs) on a chunk of The Pile

### Set up: 

Download validation data as follows, change config data_path_. E.g., fot The Pile: 
```bash
wget https://the-eye.eu/public/AI/pile/val.jsonl.zst
```

### Run code

Compute token average probabilities (optional) and intermediate outputs for calculating entropy, perplexity (e.g. logprobs) #### .zst
```bash
python scripts/compute_data.py \
    --model_config_path preset_configs/pythia70m-base.json \
    --data_path data/val.jsonl \
    --calculation_output_path outputs/pythia70m-base_calculation_data.p \
    --calc_confidence True \
    --calc_probs True \
    --max_docs 20000
```

Use intermediate outputs to compute entropy and perplexity
```bash
python scripts/calc_entropy_perplexity.py \
    --computation_data_path outputs/pythia70m-sft-sft_calculation_data_probs50000.p \
    --perplexity_output_path outputs/pythia70m-sft-perplexity.json
```

Results are visualised in notebooks/Pile_Stats_Probs.ipynb

### Evals

LIMA_gen.py computes responses for selected model on LIMA test. 
```bash
python scripts/LIMA_gen.py --model_name lomahony/eleuther-pythia2.8b-hh-sft --output_fn outputs/pythia2.8b-hh-sft_outputs_temp0.7.jsonl
```

Use gpt4_eval.py to get GPT-4 to select a winner. 

