"""
Generate hidden states of desired pythia base or finetuned model on toxic and non-toxic data points. 

Parameters: 

hf_model: str, 
n: int, 
layers = list, 

Example usage

python scripts/jigsaw_toxic_hs.py --hf_model lomahony/eleuther-pythia2.8b-hh-dpo --n 1000 --layers=[-1,-2,-3,-4,-5]

"""

import numpy as np
import pandas as pd
import os
import time
import torch
from transformers import AutoTokenizer, GPTNeoXForCausalLM # AutoModelForCausalLM, 
import fire

import jigsaw_toxic_utils


def main( hf_model: str, n: int, layers = list, ):
    jigsaw_toxic_dataset = pd.read_csv("data/jigsaw_toxic/train.csv") 
    toxic = [jigsaw_toxic_dataset[jigsaw_toxic_dataset['toxic']==1]['comment_text'][i:i+1].to_string(index=False) for i in range(0,n)]
    non_toxic = [jigsaw_toxic_dataset[jigsaw_toxic_dataset['toxic']==0]['comment_text'][i:i+1].to_string(index=False) for i in range(0,n)]

    hf_model = hf_model 
    model_name = os.path.split(os.path.split(hf_model)[1])[1]
    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    model = GPTNeoXForCausalLM.from_pretrained(hf_model, device_map="auto")

    for layer in layers:
        SAVEFOLD0 = f"outputs/{model_name}"
        SAVEFOLD = f"{SAVEFOLD0}/layer{layer}/"

        if not os.path.exists(SAVEFOLD0):
            os.mkdir(SAVEFOLD0)
        if not os.path.exists(SAVEFOLD):
            os.mkdir(SAVEFOLD)

        toxic_f = f"{SAVEFOLD}toxic_hs.npy"
        non_toxic_f = f"{SAVEFOLD}non_toxic_hs.npy"

        start_time = time.time()

        if os.path.exists(toxic_f):
            pass # toxic_hs = np.load(toxic_f, mmap_mode = 'r')
        else:
            toxic_hs = jigsaw_toxic_utils.get_hidden_states_many(model, tokenizer, toxic, n, layer, model_type="decoder")
            np.save(toxic_f, toxic_hs)

        if os.path.exists(non_toxic_f):
            pass # non_toxic_hs = np.load(non_toxic_f, mmap_mode = 'r')
        else:
            non_toxic_hs = jigsaw_toxic_utils.get_hidden_states_many(model, tokenizer, non_toxic, n, layer, model_type="decoder")
            np.save(non_toxic_f, non_toxic_hs)
            print(f"Calculated {n} toxic and non-toxic hidden states in {time.time() - start_time} seconds")


if __name__ == "__main__":
    fire.Fire(main)