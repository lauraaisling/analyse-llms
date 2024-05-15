# analyse-llms

Analysis of the effects of fine-tuning Pythia models on diversity. 

ICLR 2024 Workshop ME-FoMo paper [Attributing Mode Collapse in the Fine-Tuning of Large Language Models](https://openreview.net/forum?id=3pDMYjpOxk)

The commands to run each experiment are documented in run_log.sh. 

There are a few notebooks to visualise script output as well as some random mode collapse experiments. 

## Set up: 

Set up necessary directories: (results, results/perplexity, results/provs_conf, results/completions, results/creative_factual already exist) 
```bash
mkdir data
mkdir outputs 
```

Create and activate venv, install requirements
```bash
python3.10 -m venv analyse-llms
source analyse-llms/bin/activate
```
manually install stuff in requirements.txt

To run next token diversity metrics: 

Download the Pile validation data as follows, change config data_path_. E.g., for The Pile: 
```bash
wget https://the-eye.eu/public/AI/pile/val.jsonl.zst
```

To run: output diversity metrics: 

Download some functions for other diversity calculations by cloning https://github.com/CarperAI/diversity_metrics.git into /scripts. 
```
git clone https://github.com/CarperAI/diversity_metrics.git
```

Note: Location of Pile has changed, Pile-v2 now. 

## Running all experiments

### Running next token diversity metrics

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
python scripts/calc_perplexity.py \
    --computation_data_path outputs/pythia70m-sft-sft_calculation_data_probs50000.p \
    --perplexity_output_path outputs/perplexity/pythia70m-sft-perplexity.json
```

Results are visualised and saved by running: 
```bash
python scripts/save_results.py \
    --data_path "outputs/pythia70m-base-calculation_data10000pc.p,outputs/pythia70m-sft-calculation_data10000pc.p,outputs/pythia70m-dpo-calculation_data10000pc.p" \
    --labels "PLM,SFT,DPO" --txt_label "70m"
```

### Running output diversity metrics

Calculate completions for a model (for given prompts and a range of temperatures) by setting the model name manually in script creative_and_factual_completions.py. 
Run script: 
```bash
python scripts/creative_and_factual_completions.py
```

With these model completions, calculate the output diversity metrics by running: 
```bash
python scripts/creative_and_factual_metrics.py \
    --model "pythia-2.8b" --factual_completion_path "results/completions/pythia-2.8b_completions_factual_max_length70.npy" \
    --creative_completion_path "results/completions/pythia-2.8b_completions_creative_max_length70.npy" \
    --max_num_words 5
```

If you find the code or models useful, please feel consider citing us: 
<pre>
@inproceedings{o2024attributing,
  title={Attributing Mode Collapse in the Fine-Tuning of Large Language Models},
  author={O’Mahony, Laura and Grinsztajn, Leo and Schoelkopf, Hailey and Biderman, Stella},
  booktitle={ICLR 2024, Mathematical and Empirical Understanding of Foundation Models (ME-FoMo) workshop},
  year={2024}
}
</pre>

## Contact

For questions, please contact Laura O'Mahony - lauraaisling.ml@gmail.com
