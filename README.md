# analyse-llms

Exploratory analysis of the effects of fine tuning pythia models. 

There are several notebooks to visualise script output as well as some random mode collapse experiments. 

## To compute token average probabilities (optional) and intermediate outputs for calculating entropy, perplexity (e.g. logprobs)

### Set up: 

Download The Pile validation data as follows: 
```bash
wget https://the-eye.eu/public/AI/pile/val.jsonl.zst
```

### Run code

Compute token average probabilities (optional) and intermediate outputs for calculating entropy, perplexity (e.g. logprobs)
```bash
python scripts/compute_data.py \
    --model_config_path preset_configs/pythia70m-sft.json \
    --data_path data/val.jsonl.zst \
    --calculation_output_path outputs/pythia70m-sft_calculation_data_probs50000.p \
    --calc_probs True
    --max_docs 200
```

Use intermediate outputs to compute entropy and perplexity
```bash
python scripts/calc_entropy_perplexity.py \
    --computation_data_path outputs/pythia70m-sft-sft_calculation_data_probs50000.p \
    --perplexity_output_path outputs/pythia70m-sft-perplexity.json
```

Results are visualised in notebooks/Pile_Stats_Probs.ipynb


