from functools import partial
import torch
from torch.nn import CosineSimilarity
import numpy as np
import argparse
import sys
import os
sys.path.append('./scripts/') # /scripts

# import metrics from https://github.com/CarperAI/diversity_metrics/tree/main

from sentence_transformers import SentenceTransformer
from diversity_metrics.metrics.model_free_metrics import *
from diversity_metrics.embeddings.models import *
from diversity_metrics.metrics.generalized_diversity import *

from vendi_score import vendi
# import combinations
from itertools import combinations
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

parser = argparse.ArgumentParser(description='Arguments for run')
parser.add_argument("--model", help="Name of model to save under correct name", type=str, required=True)
parser.add_argument("--revision", help="Revision/epoch of model to save under correct name", type=str, default="main")
parser.add_argument("--seen_completion_path", help="Paths to seen completions calculated in creative_and_factual_completions.py (.np files)", type=str, required=True)
parser.add_argument("--max_num_words", default=20, type=int)

args = parser.parse_args()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

def limit_num_words(sentence, max_num_words):
    return " ".join(sentence.split()[:max_num_words])


def average_pairwise_jaccard(sentences, n=2):
    return np.mean([pairwise_ngram(n, x, y) for x, y in combinations(sentences, 2)])

def self_bleu_smooth(sentences):
    '''
    Calculates the Self-BLEU score for a collection of generated examples (https://arxiv.org/abs/1802.01886)
    :param sentences: List of generated examples
    :return:
    '''

    scores = []
    for i, hypothesis in enumerate(sentences):
        hypothesis_split = hypothesis.strip().split()

        references = [sentences[j].strip().split() for j in range(len(sentences)) if i != j]

        scores.append(sentence_bleu(references, hypothesis_split, smoothing_function=SmoothingFunction().method1))

    return sum(scores) / len(scores)

diversity_metrics = {"selfBleuSmoothed": self_bleu_smooth,
                    "average_pairwise_ncd": lambda sentences: np.mean(get_pairwise_ncd(sentences)),
                     "average_pairwise_jaccard_2": partial(average_pairwise_jaccard, n=2),
                    "average_pairwise_jaccard_3": partial(average_pairwise_jaccard, n=3),
                    "average_pairwise_jaccard_4": partial(average_pairwise_jaccard, n=4),
                    "avg_compression_ratio_full": avg_compression_ratio_full,
                    "avg_compression_ratio_target": avg_compression_ratio_target,
                    "cosine_similarity": None # will be filled in later
                    }

pairwise_similarities = {"jaccard_2": partial(pairwise_ngram, 2), "cosine": None}
qs = [2,3,4,5,6]
num_samples = 50
def generate_metric_order(q, metric):
    return lambda sentences: diversity_order_q(sentences, q, metric, num_samples) # careful about lambda in loops
for q in qs:
    for key, metric in pairwise_similarities.items():
        diversity_metrics[f"order_{q}_{key}"] = generate_metric_order(q, metric)


def generate_metric_vendi(metric):
    return lambda sentences: vendi.score(sentences, metric)
for key, metric in pairwise_similarities.items():
    diversity_metrics[f"vendi_{key}"] = generate_metric_vendi(metric)

def compute_metric(key, data, prompt_type):
    if "cosine" in key:
        sent_embedder = SBERTEmbedder()
        sent_embedder.model = sent_embedder.model.to(device)
        metric_func = partial(get_pairwise_cosine_sim, sent_embedder)
        if key == "cosine_similarity":
            metric = partial(get_avg_cosine_sim, sent_embedder)
        elif key.startswith("order_") and "cosine" in key:
            q = int(key.split("_")[1])
            metric_func = partial(get_pairwise_cosine_sim, sent_embedder)
            metric = lambda sentences: diversity_order_q(sentences, q, metric_func, num_samples)
        elif key.startswith("vendi_") and "cosine" in key:
            metric_func = partial(get_pairwise_cosine_sim, sent_embedder)
            metric = lambda sentences: vendi.score(sentences, metric_func)
    else:
        metric = diversity_metrics[key]

    result = np.zeros_like(data, dtype=np.float32)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):
                try:
                    result[i, j, k] = metric(data[i, j, k])
                except:
                    result[i, j, k] = np.nan
                print(result[i, j, k])
    return key, result, prompt_type

completions_seen = np.load(args.seen_completion_path, allow_pickle=True)

max_num_words = args.max_num_words # 20, 5 

results = {"seen": {}}
for key, metric in diversity_metrics.items():
    for data_full in [completions_seen]:
        # truncate the sentences to max_num_words
        data = np.zeros_like(data_full, dtype=object)
        for i in range(data_full.shape[0]): # metrics
            for j in range(data_full.shape[1]): # temps
                for k in range(data_full.shape[2]): # prompts
                    data[i, j, k] = list(map(lambda sentence: limit_num_words(sentence, max_num_words), data_full[i, j, k]))

        if data_full is completions_seen: 
            metric_name, result_data, data_type = compute_metric(key, data, "seen")
            results[data_type][metric_name] = result_data

if not os.path.exists('results/seen'):
    os.mkdir('results/seen')
np.save(f'results/seen/{args.model}_{args.revision}_results_{max_num_words}_words.npy', results)