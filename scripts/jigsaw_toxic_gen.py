"""
Generate hidden states of desired pythia base or finetuned model on toxic and non-toxic data points. 

Parameters: 

model_name: str, 
output_fn: str, 
n: int, 
layers = list, 

Example usage

python scripts/jigsaw_toxic_gen.py --model_name lomahony/pythia-2.8b-helpful-sft --output_fn outputs/pythia-2.8b-helpful-sft-outputs_jigsaw_toxic_100_temp0.7.jsonl

python scripts/jigsaw_toxic_gen.py --model_name EleutherAI/pythia-2.8b --output_fn outputs/pythia2.8b-outputs_jigsaw_toxic_100_temp0.7.jsonl

"""

import json
import time
# from datasets import load_dataset
import pandas as pd
import torch
from transformers import AutoTokenizer, GPTNeoXForCausalLM # AutoModel, AutoModelForCausalLM, 
import fire

# B_INST, E_INST = "[INST]", "[/INST]"
# B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
system_prompt = ""
# DEFAULT_SYSTEM_PROMPT = """\
# You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

# If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""


def decode_wrapper(prompt, tokenizer, model, max_new_tokens=1000, temperature=0.7, top_p=1.0):
    # formatted_prompt = f"{B_INST} {B_SYS}{DEFAULT_SYSTEM_PROMPT}{E_SYS}{prompt.strip()} {E_INST}" # llama2-chat specific format
    formatted_prompt = f"{system_prompt}{prompt.strip()}"
    with torch.no_grad():
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(inputs=inputs.input_ids, pad_token_id=tokenizer.eos_token_id, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, do_sample=True)
        output_ids = outputs[0][len(inputs.input_ids[0]):].tolist()

        end_idx = len(output_ids)
        processed_string = tokenizer.decode(output_ids[:end_idx], skip_special_tokens=True).strip()
        
    return processed_string


def main( model_name: str, output_fn: str,):
    jigsaw_toxic_dataset = pd.read_csv("data/jigsaw_toxic/train.csv") 
    jigsaw_toxic_dataset = jigsaw_toxic_dataset[:100] # try 100 for now

    major, minor = torch.cuda.get_device_capability()
    if major >= 8:
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float16

    model_name = model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = GPTNeoXForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype, device_map="auto")

    generations = []
    prompt_list = [ e for e in jigsaw_toxic_dataset[jigsaw_toxic_dataset['toxic']==1]['comment_text'] ]
    repeat = 1
    start_time = time.time()
    for i, prompt in enumerate(prompt_list):
        # print(i)
        if i<20: 
            generations.append({'prompt': prompt, 'response_list': [decode_wrapper(prompt, tokenizer, model) for _ in range(repeat)]})
            if (i+1) % 10 == 0:
                print(f"finished {i+1} prompts in {time.time() - start_time} seconds")
        # else: 
        #     print(f"finished {i+1} prompts in {time.time() - start_time} seconds")

    with open(output_fn, "w") as f:
        for gen in generations:
            f.write(json.dumps(gen))
            f.write('\n')


if __name__ == "__main__":
    fire.Fire(main)