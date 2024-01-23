# import libraries
import sys
sys.path.append('./outputs/')
import torch
import numpy as np
import json
import os
from matplotlib import pyplot as plt
import argparse

torch.cuda.is_available()

parser = argparse.ArgumentParser(description='Arguments for run')
parser.add_argument("--data_path", help="Paths to compute_data.py outputs (.p files)", type=str, required=True)
# parser.add_argument("--perplexity_path", help="Paths to calc_entropy_perplexity.py outputs (json files)", type=str, required=True)
parser.add_argument("--labels", help="labels for plots... ", type=str, required=True)
parser.add_argument("--txt_label", help="label for output txt and png files... ", type=str, required=True)

args = parser.parse_args()

# plt.rcParams["figure.figsize"] = (3,3)

SAVEFOLD=f"results/"
output_f = f"{SAVEFOLD}probs_{args.txt_label}.txt" 

# https://stackoverflow.com/questions/15753701/how-can-i-pass-a-list-as-a-command-line-argument-with-argparse 
labels = [item for item in args.labels.split(',')]

# data_path = args.data_path
# calc_data = torch.load(data_path)
calc_datas = [torch.load(item) for item in args.data_path.split(',')]

# perplexity_path = f'{args.perplexity_path}'
# with open(perplexity_path,'r') as file: 
#     perplexity_data = json.loads(file.read())
# perplexity_data = []
# perplexity_paths = [item for item in args.perplexity_path.split(',')] # f'{args.perplexity_path}'
# for i in range(len(perplexity_paths)):
#     with open(perplexity_paths,'r') as file: 
#         perplexity_data.append(json.loads(file.read())) 

# entropy
probs = []
probs_list = []
probs_order = []
probs_list_sorted = []
cdfs = []
for i in range(len(calc_datas)):
    with open(output_f, 'a') as fp:
        fp.write(f"Entropy {labels[i]}:  {calc_datas[i]['entropy'][0].mean()} \n")
    # calc_datas[i]["entropy"][0].mean()
    probs.append(np.concatenate(calc_datas[i]["mean_probs"]).mean(0))
    probs_list.append(list(probs[-1]))
    probs_order.append(np.argsort(probs_list[-1]))
    probs_list_sorted.append(sorted(probs_list[-1], reverse=True))
    cdfs.append(np.cumsum(probs_list_sorted[-1])) #############
with open(output_f, 'a') as fp:
    fp.write("\n")

full_f = f"{SAVEFOLD}cdf_{args.txt_label}.png"
for i in range(len(cdfs)): 
    plt.plot(cdfs[i], label=labels[i])
plt.title("CDF of average token probability on The Pile validation")
plt.xlabel("Number of tokens")
plt.ylabel("Cumulative average probability")
plt.legend()
plt.savefig(full_f, bbox_inches="tight") 
plt.close()

zoom_f = f"{SAVEFOLD}cdf_{args.txt_label}-zoom.png"
for i in range(len(cdfs)): 
    plt.plot(cdfs[i], label=labels[i])
plt.title("CDF of average token probability on The Pile validation")
plt.xlabel("Number of tokens")
plt.ylabel("Cumulative average probability")
plt.legend()
plt.ylim((0.8,1))
plt.savefig(zoom_f, bbox_inches="tight") 
plt.close()

def first(the_iterable, condition = lambda x: True):
    for idx, cumprob in enumerate(the_iterable):
        if condition(cumprob):
            return idx

for i in range(len(cdfs)): 
    # 50304 tokens in total. 
    tot_toks = len(cdfs[i])

    # print("Proportion of total tokenizer where sum of average token probabilities at 70%: ")
    # print(f"{first(cdfs[i], lambda i: i > 0.7)/tot_toks}")

    # print("Proportion of total tokenizer where sum of average token probabilities at 90%: ")
    # print(f"{first(cdfs[i], lambda i: i > 0.9)/tot_toks}")
    with open(output_f, 'a') as fp:
        fp.write(f"Proportion of total tokenizer where sum of average token probabilities at 70th percentile for {labels[i]}:  {first(cdfs[i], lambda i: i > 0.7)/tot_toks} \n")
        fp.write(f"Proportion of total tokenizer where sum of average token probabilities at 90th percentile for {labels[i]}:  {first(cdfs[i], lambda i: i > 0.9)/tot_toks} \n \n \n ")

