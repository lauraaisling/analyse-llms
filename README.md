# analyse-llms

```bash
# Compute intermediate outputs for calculating perplexity (e.g. logprobs)
python scripts/calculation_data.py \
    --model_config_path preset_configs/pythia70m-sft.json \
    --data_path data/val.jsonl.zst \
    --calculation_output_path outputs/pythia70m-sft/calculation_data.p
```

# Use intermediate outputs to compute perplexity
```bash
python scripts/compute_perplexity.py \
    --calculation_output_path outputs/pythia70m-sft-calculation_data.p \
    --perplexity_output_path outputs/pythia70m-sft-perplexity.json
```

