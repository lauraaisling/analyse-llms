# analyse-llms

Compute token average probabilities (optional) and intermediate outputs for calculating entropy, perplexity (e.g. logprobs)
```bash
python scripts/compute_data.py \
    --model_config_path preset_configs/pythia70m-sft.json \
    --data_path data/val.jsonl.zst \
    --calculation_output_path outputs/pythia70m-sft_calculation_data_probs50000.p \
    --calc_probs True
```

Use intermediate outputs to compute perplexity
```bash
python scripts/calc_entropy_perplexity.py \
    --computation_data_path outputs/pythia70m-sft-sft_calculation_data_probs50000.p \
    --perplexity_output_path outputs/pythia70m-sft-perplexity.json
```


Set up: 

Download The Pile validation data as follows: 
```bash
wget https://the-eye.eu/public/AI/pile/val.jsonl.zst
```

## My notes: 
Ran: 
```bash
python scripts/compute_data.py --model_config_path preset_configs/pythia70m-base.json --data_path data/val.jsonl.zst --calculation_output_path outputs/pythia70m-base_calculation_data_probs50000.p
```
and sft, dpo. 

```bash
python scripts/compute_perplexity.py --perplexity_data_path outputs/pythia70m-base_calculation_data_probs50000.p --output_path outputs/pythia70m-base-50000.json
```

and sft, dpo. 
