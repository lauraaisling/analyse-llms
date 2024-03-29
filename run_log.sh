############# PERPLEXITY!!! ##################

echo 'perplexity calcs 70m'
python scripts/compute_data.py --model_config_path preset_configs/pythia70m-base.json --data_path data/val.jsonl --calculation_output_path outputs/pythia70m-base-calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000
python scripts/compute_data.py --model_config_path preset_configs/pythia70m-sft.json --data_path data/val.jsonl --calculation_output_path outputs/pythia70m-sft-calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000
python scripts/compute_data.py --model_config_path preset_configs/pythia70m-dpo.json --data_path data/val.jsonl --calculation_output_path outputs/pythia70m-dpo-calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000
python scripts/compute_data.py --model_config_path preset_configs/pythia70m-sft-epoch2.json --data_path data/val.jsonl --calculation_output_path outputs/pythia70m-sft-epoch2-calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000
echo 'perplexity 70m'
python scripts/calc_perplexity.py --calculation_output_path outputs/pythia70m-base-calculation_data10000pc.p --output_path results/perplexity/pythia70m-base-perplexity.json
python scripts/calc_perplexity.py --calculation_output_path outputs/pythia70m-sft-calculation_data10000pc.p --output_path results/perplexity/pythia70m-sft-perplexity.json
python scripts/calc_perplexity.py --calculation_output_path outputs/pythia70m-dpo-calculation_data10000pc.p --output_path results/perplexity/pythia70m-dpo-perplexity.json
python scripts/calc_perplexity.py --calculation_output_path outputs/pythia70m-sft-epoch2-calculation_data10000pc.p --output_path results/perplexity/pythia70m-sft-epoch2-perplexity.json

echo 'perplexity calcs 160m'
python scripts/compute_data.py --model_config_path preset_configs/pythia160m-base.json --data_path data/val.jsonl --calculation_output_path outputs/pythia160m-base-calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000
python scripts/compute_data.py --model_config_path preset_configs/pythia160m-sft.json --data_path data/val.jsonl --calculation_output_path outputs/pythia160m-sft-calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000
python scripts/compute_data.py --model_config_path preset_configs/pythia160m-dpo.json --data_path data/val.jsonl --calculation_output_path outputs/pythia160m-dpo-calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000
python scripts/compute_data.py --model_config_path preset_configs/pythia160m-sft-epoch2.json --data_path data/val.jsonl --calculation_output_path outputs/pythia160m-sft-epoch2-calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000
echo 'perplexity 160m'
python scripts/calc_perplexity.py --calculation_output_path outputs/pythia160m-base-calculation_data10000pc.p --output_path results/perplexity/pythia160m-base-perplexity.json
python scripts/calc_perplexity.py --calculation_output_path outputs/pythia160m-sft-calculation_data10000pc.p --output_path results/perplexity/pythia160m-sft-perplexity.json
python scripts/calc_perplexity.py --calculation_output_path outputs/pythia160m-dpo-calculation_data10000pc.p --output_path results/perplexity/pythia160m-dpo-perplexity.json
python scripts/calc_perplexity.py --calculation_output_path outputs/pythia160m-sft-epoch2-calculation_data10000pc.p --output_path results/perplexity/pythia160m-sft-epoch2-perplexity.json

echo 'perplexity calcs 410m'
python scripts/compute_data.py --model_config_path preset_configs/pythia410m-base.json --data_path data/val.jsonl --calculation_output_path outputs/pythia410m-base-calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000
python scripts/compute_data.py --model_config_path preset_configs/pythia410m-sft.json --data_path data/val.jsonl --calculation_output_path outputs/pythia410m-sft-calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000
python scripts/compute_data.py --model_config_path preset_configs/pythia410m-dpo.json --data_path data/val.jsonl --calculation_output_path outputs/pythia410m-dpo-calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000
python scripts/compute_data.py --model_config_path preset_configs/pythia410m-sft-epoch2.json --data_path data/val.jsonl --calculation_output_path outputs/pythia410m-sft-epoch2-calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000
echo 'perplexity 410m'
python scripts/calc_perplexity.py --calculation_output_path outputs/pythia410m-base-calculation_data10000pc.p --output_path results/perplexity/pythia410m-base-perplexity.json
python scripts/calc_perplexity.py --calculation_output_path outputs/pythia410m-sft-calculation_data10000pc.p --output_path results/perplexity/pythia410m-sft-perplexity.json
python scripts/calc_perplexity.py --calculation_output_path outputs/pythia410m-dpo-calculation_data10000pc.p --output_path results/perplexity/pythia410m-dpo-perplexity.json
python scripts/calc_perplexity.py --calculation_output_path outputs/pythia410m-sft-epoch2-calculation_data10000pc.p --output_path results/perplexity/pythia410m-sft-epoch2-perplexity.json

echo 'perplexity calcs 1b'
python scripts/compute_data.py --model_config_path preset_configs/pythia1b-base.json --data_path data/val.jsonl --calculation_output_path outputs/pythia1b-base-calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000
python scripts/compute_data.py --model_config_path preset_configs/pythia1b-sft.json --data_path data/val.jsonl --calculation_output_path outputs/pythia1b-sft-calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000
python scripts/compute_data.py --model_config_path preset_configs/pythia1b-dpo.json --data_path data/val.jsonl --calculation_output_path outputs/pythia1b-dpo-calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000
python scripts/compute_data.py --model_config_path preset_configs/pythia1b-sft-epoch2.json --data_path data/val.jsonl --calculation_output_path outputs/pythia1b-sft-epoch2-calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000
echo 'perplexity 1b'
python scripts/calc_perplexity.py --calculation_output_path outputs/pythia1b-base-calculation_data10000pc.p --output_path results/perplexity/pythia1b-base-perplexity.json
python scripts/calc_perplexity.py --calculation_output_path outputs/pythia1b-sft-calculation_data10000pc.p --output_path results/perplexity/pythia1b-sft-perplexity.json
python scripts/calc_perplexity.py --calculation_output_path outputs/pythia1b-dpo-calculation_data10000pc.p --output_path results/perplexity/pythia1b-dpo-perplexity.json
python scripts/calc_perplexity.py --calculation_output_path outputs/pythia1b-sft-epoch2-calculation_data10000pc.p --output_path results/perplexity/pythia1b-sft-epoch2-perplexity.json

echo 'perplexity calcs 1.4b'
python scripts/compute_data.py --model_config_path preset_configs/pythia14b-base.json --data_path data/val.jsonl --calculation_output_path outputs/pythia14b-base-calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000
python scripts/compute_data.py --model_config_path preset_configs/pythia14b-sft.json --data_path data/val.jsonl --calculation_output_path outputs/pythia14b-sft-calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000
python scripts/compute_data.py --model_config_path preset_configs/pythia14b-dpo.json --data_path data/val.jsonl --calculation_output_path outputs/pythia14b-dpo-calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000
python scripts/compute_data.py --model_config_path preset_configs/pythia14b-sft-epoch2.json --data_path data/val.jsonl --calculation_output_path outputs/pythia14b-sft-epoch2-calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000
echo 'perplexity 1.4b'
python scripts/calc_perplexity.py --calculation_output_path outputs/pythia14b-base-calculation_data10000pc.p --output_path results/perplexity/pythia14b-base-perplexity.json
python scripts/calc_perplexity.py --calculation_output_path outputs/pythia14b-sft-calculation_data10000pc.p --output_path results/perplexity/pythia14b-sft-perplexity.json
python scripts/calc_perplexity.py --calculation_output_path outputs/pythia14b-dpo-calculation_data10000pc.p --output_path results/perplexity/pythia14b-dpo-perplexity.json
python scripts/calc_perplexity.py --calculation_output_path outputs/pythia14b-sft-epoch2-calculation_data10000pc.p --output_path results/perplexity/pythia14b-sft-epoch2-perplexity.json

echo 'perplexity calcs 2.8b'
python scripts/compute_data.py --model_config_path preset_configs/pythia28b-base.json --data_path data/val.jsonl --calculation_output_path outputs/pythia28b-base-calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000
python scripts/compute_data.py --model_config_path preset_configs/pythia28b-sft.json --data_path data/val.jsonl --calculation_output_path outputs/pythia28b-sft-calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000
python scripts/compute_data.py --model_config_path preset_configs/pythia28b-dpo.json --data_path data/val.jsonl --calculation_output_path outputs/pythia28b-dpo-calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000
python scripts/compute_data.py --model_config_path preset_configs/pythia28b-sft-epoch2.json --data_path data/val.jsonl --calculation_output_path outputs/pythia28b-sft-epoch2-calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000
echo 'perplexity 2.8b'
python scripts/calc_perplexity.py --calculation_output_path outputs/pythia28b-base-calculation_data10000pc.p --output_path results/perplexity/pythia28b-base-perplexity.json
python scripts/calc_perplexity.py --calculation_output_path outputs/pythia28b-sft-calculation_data10000pc.p --output_path results/perplexity/pythia28b-sft-perplexity.json
python scripts/calc_perplexity.py --calculation_output_path outputs/pythia28b-dpo-calculation_data10000pc.p --output_path results/perplexity/pythia28b-dpo-perplexity.json
python scripts/calc_perplexity.py --calculation_output_path outputs/pythia28b-sft-epoch2-calculation_data10000pc.p --output_path results/perplexity/pythia28b-sft-epoch2-perplexity.json

# echo 'perplexity calcs 6.9b'
# python scripts/compute_data.py --model_config_path preset_configs/pythia69b-base.json --data_path data/val.jsonl --calculation_output_path outputs/pythia69b-base-calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000
# python scripts/compute_data.py --model_config_path preset_configs/pythia69b-sft.json --data_path data/val.jsonl --calculation_output_path outputs/pythia69b-sft-calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000
# python scripts/compute_data.py --model_config_path preset_configs/pythia69b-dpo.json --data_path data/val.jsonl --calculation_output_path outputs/pythia69b-dpo-calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000
# python scripts/compute_data.py --model_config_path preset_configs/pythia69b-ppo.json --data_path data/val.jsonl --calculation_output_path outputs/pythia69b-ppo-calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000
# echo 'perplexity 6.9b'
# python scripts/calc_perplexity.py --calculation_output_path outputs/pythia69b-base-calculation_data10000pc.p --output_path results/perplexity/pythia69b-base-perplexity.json
# python scripts/calc_perplexity.py --calculation_output_path outputs/pythia69b-sft-calculation_data10000pc.p --output_path results/perplexity/pythia69b-sft-perplexity.json
# python scripts/calc_perplexity.py --calculation_output_path outputs/pythia69b-dpo-calculation_data10000pc.p --output_path results/perplexity/pythia69b-dpo-perplexity.json
# python scripts/calc_perplexity.py --calculation_output_path outputs/pythia69b-ppo-calculation_data10000pc.p --output_path results/perplexity/pythia69b-ppo-perplexity.json

echo 'perplexity calcs llama'
python scripts/compute_data.py --model_config_path preset_configs/llama-7b.json --data_path data/val.jsonl --calculation_output_path outputs/llama-7b-base_calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000
python scripts/compute_data.py --model_config_path preset_configs/llama-7b-chat.json --data_path data/val.jsonl --calculation_output_path outputs/llama-7b-chat_calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000
echo 'perplexity llama'
python scripts/calc_perplexity.py --calculation_output_path outputs/llama-7b-base_calculation_data10000pc.p --output_path results/perplexity/llama-7b-perplexity.json
python scripts/calc_perplexity.py --calculation_output_path outputs/llama-7b-chat_calculation_data10000pc.p --output_path results/perplexity/llama-7b-chat-perplexity.json

echo 'perplexity calcs llama finetuned'
python scripts/compute_data.py --model_config_path preset_configs/llama-7b-dpo.json --data_path data/val.jsonl --calculation_output_path outputs/llama-7b-dpo_calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000
python scripts/compute_data.py --model_config_path preset_configs/llama-7b-ppo.json --data_path data/val.jsonl --calculation_output_path outputs/llama-7b-ppo_calculation_data10000pc.p --calc_confidence True --calc_probs True --max_docs 10000
echo 'perplexity llama finetuned'
python scripts/calc_perplexity.py --calculation_output_path outputs/llama-7b-dpo_calculation_data10000pc.p --output_path results/perplexity/llama-7b-dpo-perplexity.json
python scripts/calc_perplexity.py --calculation_output_path outputs/llama-7b-sft_calculation_data10000pc.p --output_path results/perplexity/llama-7b-sft-perplexity.json
python scripts/calc_perplexity.py --calculation_output_path outputs/llama-7b-ppo_calculation_data10000pc.p --output_path results/perplexity/llama-7b-ppo-perplexity.json


echo 'save_results'
python scripts/save_results.py --data_path "outputs/pythia70m-base-calculation_data10000pc.p,outputs/pythia70m-sft-calculation_data10000pc.p,outputs/pythia70m-dpo-calculation_data10000pc.p" --labels "PLM,SFT,DPO" --txt_label "70m"
python scripts/save_results.py --data_path "outputs/pythia160m-base-calculation_data10000pc.p,outputs/pythia160m-sft-calculation_data10000pc.p,outputs/pythia160m-dpo-calculation_data10000pc.p" --labels "PLM,SFT,DPO" --txt_label "160m"
python scripts/save_results.py --data_path "outputs/pythia410m-base-calculation_data10000pc.p,outputs/pythia410m-sft-calculation_data10000pc.p,outputs/pythia410m-dpo-calculation_data10000pc.p" --labels "PLM,SFT,DPO" --txt_label "410m"
python scripts/save_results.py --data_path "outputs/pythia1b-base-calculation_data10000pc.p,outputs/pythia1b-sft-calculation_data10000pc.p,outputs/pythia1b-dpo-calculation_data10000pc.p" --labels "PLM,SFT,DPO" --txt_label "1b"
python scripts/save_results.py --data_path "outputs/pythia14b-base-calculation_data10000pc.p,outputs/pythia14b-sft-calculation_data10000pc.p,outputs/pythia14b-dpo-calculation_data10000pc.p" --labels "PLM,SFT,DPO" --txt_label "1.4b"
python scripts/save_results.py --data_path "outputs/pythia28b-base-calculation_data10000pc.p,outputs/pythia28b-sft-calculation_data10000pc.p,outputs/pythia28b-dpo-calculation_data10000pc.p" --labels "PLM,SFT,DPO" --txt_label "2.8b"
# python scripts/save_results.py --data_path "outputs/pythia69b-base-calculation_data10000pc.p,outputs/pythia69b-sft-calculation_data10000pc.p,outputs/pythia69b-dpo-calculation_data10000pc.p,outputs/pythia69b-ppo-calculation_data10000pc.p" --labels "PLM,SFT,DPO,PPO" --txt_label "6.9b"
python scripts/save_results.py --data_path "outputs/llama-7b-base_calculation_data10000pc.p,outputs/llama-7b-chat_calculation_data10000pc.p" --labels "PLM,chat" --txt_label "llama"
# python scripts/save_results.py --data_path "outputs/pythia69b-base-calculation_data10000pc.p,outputs/llama-7b-base_calculation_data10000pc.p,outputs/llama-7b-chat_calculation_data10000pc.p" --labels "pythia-6.9b,llama-2-7b,llama-2-7b-chat" --txt_label "pythia_base_llama"
python scripts/save_results.py --data_path "outputs/llama-7b-sft_calculation_data10000pc.p,outputs/llama-7b-dpo_calculation_data10000pc.p,outputs/llama-7b-ppo_calculation_data10000pc.p" --labels "SFT,DPO,PPO" --txt_label "llama-contextualAI"

python scripts/save_results.py --data_path "outputs/pythia70m-sft-epoch2-calculation_data10000pc.p" --labels "SFT2" --txt_label "70m2"
python scripts/save_results.py --data_path "outputs/pythia160m-sft-epoch2-calculation_data10000pc.p" --labels "SFT2" --txt_label "160m2"
python scripts/save_results.py --data_path "outputs/pythia410m-sft-epoch2-calculation_data10000pc.p" --labels "SFT2" --txt_label "410m2"
python scripts/save_results.py --data_path "outputs/pythia1b-sft-epoch2-calculation_data10000pc.p" --labels "SFT2" --txt_label "1b2"
python scripts/save_results.py --data_path "outputs/pythia14b-sft-epoch2-calculation_data10000pc.p" --labels "SFT2" --txt_label "1.4b2"
python scripts/save_results.py --data_path "outputs/pythia28b-sft-epoch2-calculation_data10000pc.p" --labels "SFT2" --txt_label "2.8b2"

############# CREATIVE AND FACTUAL Completions!!! ##################

# change model name manually in script
python scripts/creative_and_factual_completions.py

############# CREATIVE AND FACTUAL METRICS!!! ##################

python scripts/creative_and_factual_metrics.py --model "pythia-2.8b" --factual_completion_path "results/completions/pythia-2.8b_completions_factual_max_length70.npy" --creative_completion_path "results/completions/pythia-2.8b_completions_creative_max_length70.npy" --max_num_words 5
python scripts/creative_and_factual_metrics.py --model "pythia-2.8b-sft" --factual_completion_path "results/completions/pythia-2.8b-sft_completions_factual_max_length70.npy" --creative_completion_path "results/completions/pythia-2.8b-sft_completions_creative_max_length70.npy" --max_num_words 5
python scripts/creative_and_factual_metrics.py --model "pythia-2.8b-dpo" --factual_completion_path "results/completions/pythia-2.8b-dpo_completions_factual_max_length70.npy" --creative_completion_path "results/completions/pythia-2.8b-dpo_completions_creative_max_length70.npy" --max_num_words 5

python scripts/creative_and_factual_metrics.py --model "pythia-2.8b" --factual_completion_path "results/completions/pythia-2.8b_completions_factual_max_length70.npy" --creative_completion_path "results/completions/pythia-2.8b_completions_creative_max_length70.npy"
python scripts/creative_and_factual_metrics.py --model "pythia-2.8b-sft" --factual_completion_path "results/completions/pythia-2.8b-sft_completions_factual_max_length70.npy" --creative_completion_path "results/completions/pythia-2.8b-sft_completions_creative_max_length70.npy"
python scripts/creative_and_factual_metrics.py --model "pythia-2.8b-dpo" --factual_completion_path "results/completions/pythia-2.8b-dpo_completions_factual_max_length70.npy" --creative_completion_path "results/completions/pythia-2.8b-dpo_completions_creative_max_length70.npy"

python scripts/creative_and_factual_metrics.py --model "llama2-chat" --factual_completion_path "results/completions/llama2-chat_completions_factual_max_length70.npy" --creative_completion_path "results/completions/llama2-chat_completions_creative_max_length70.npy" --max_num_words 5
python scripts/creative_and_factual_metrics.py --model "llama2" --factual_completion_path "results/completions/llama2_completions_factual_max_length70.npy" --creative_completion_path "results/completions/llama2_completions_creative_max_length70.npy" --max_num_words 5
# python scripts/creative_and_factual_metrics.py --model "pythia-6.9b" --factual_completion_path "results/completions/pythia-6.9b_completions_factual_max_length70.npy" --creative_completion_path "results/completions/pythia-6.9b_completions_creative_max_length70.npy" --max_num_words 5
# python scripts/creative_and_factual_metrics.py --model "pythia-6.9b-sft" --factual_completion_path "results/completions/pythia-6.9b-sft_completions_factual_max_length70.npy" --creative_completion_path "results/completions/pythia-6.9b-sft_completions_creative_max_length70.npy" --max_num_words 5
# python scripts/creative_and_factual_metrics.py --model "pythia-6.9b-dpo" --factual_completion_path "results/completions/pythia-6.9b-dpo_completions_factual_max_length70.npy" --creative_completion_path "results/completions/pythia-6.9b-dpo_completions_creative_max_length70.npy" --max_num_words 5
# python scripts/creative_and_factual_metrics.py --model "pythia-6.9b-ppo" --factual_completion_path "results/completions/pythia-6.9b-ppo_completions_factual_max_length70.npy" --creative_completion_path "results/completions/pythia-6.9b-ppo_completions_creative_max_length70.npy" --max_num_words 5
python scripts/creative_and_factual_metrics.py --model "llama2-sft" --factual_completion_path "results/completions/llama2-sft_completions_factual_max_length70.npy" --creative_completion_path "results/completions/llama2-sft_completions_creative_max_length70.npy" --max_num_words 5
python scripts/creative_and_factual_metrics.py --model "llama2-dpo" --factual_completion_path "results/completions/llama2-dpo_completions_factual_max_length70.npy" --creative_completion_path "results/completions/llama2-dpo_completions_creative_max_length70.npy" --max_num_words 5
python scripts/creative_and_factual_metrics.py --model "llama2-ppo" --factual_completion_path "results/completions/llama2-ppo_completions_factual_max_length70.npy" --creative_completion_path "results/completions/llama2-ppo_completions_creative_max_length70.npy" --max_num_words 5

python scripts/creative_and_factual_metrics.py --model "llama2-chat" --factual_completion_path "results/completions/llama2-chat_completions_factual_max_length70.npy" --creative_completion_path "results/completions/llama2-chat_completions_creative_max_length70.npy"
python scripts/creative_and_factual_metrics.py --model "llama2" --factual_completion_path "results/completions/llama2_completions_factual_max_length70.npy" --creative_completion_path "results/completions/llama2_completions_creative_max_length70.npy"
# python scripts/creative_and_factual_metrics.py --model "pythia-6.9b" --factual_completion_path "results/completions/pythia-6.9b_completions_factual_max_length70.npy" --creative_completion_path "results/completions/pythia-6.9b_completions_creative_max_length70.npy"
# python scripts/creative_and_factual_metrics.py --model "pythia-6.9b-sft" --factual_completion_path "results/completions/pythia-6.9b-sft_completions_factual_max_length70.npy" --creative_completion_path "results/completions/pythia-6.9b-sft_completions_creative_max_length70.npy"
# python scripts/creative_and_factual_metrics.py --model "pythia-6.9b-dpo" --factual_completion_path "results/completions/pythia-6.9b-dpo_completions_factual_max_length70.npy" --creative_completion_path "results/completions/pythia-6.9b-dpo_completions_creative_max_length70.npy"
# python scripts/creative_and_factual_metrics.py --model "pythia-6.9b-ppo" --factual_completion_path "results/completions/pythia-6.9b-ppo_completions_factual_max_length70.npy" --creative_completion_path "results/completions/pythia-6.9b-ppo_completions_creative_max_length70.npy"
python scripts/creative_and_factual_metrics.py --model "llama2-sft" --factual_completion_path "results/completions/llama2-sft_completions_factual_max_length70.npy" --creative_completion_path "results/completions/llama2-sft_completions_creative_max_length70.npy"
python scripts/creative_and_factual_metrics.py --model "llama2-dpo" --factual_completion_path "results/completions/llama2-dpo_completions_factual_max_length70.npy" --creative_completion_path "results/completions/llama2-dpo_completions_creative_max_length70.npy"
python scripts/creative_and_factual_metrics.py --model "llama2-ppo" --factual_completion_path "results/completions/llama2-ppo_completions_factual_max_length70.npy" --creative_completion_path "results/completions/llama2-ppo_completions_creative_max_length70.npy"

