import argparse
import torch
from tqdm import auto as tqdm_lib
import numpy as np
import lm_dataformat
import pythia_funcs 
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config_path', required=True,
                        help='Path to where model config is stored')
    parser.add_argument('--data_path', required=True,
                        help='Path to where subset of the pile is stored')
    parser.add_argument('--calculation_output_path', required=True,
                        help='Path for where to save results')
    parser.add_argument('--calc_confidence', default=False, 
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']),
                        help='Save probability of next token prediction - Yes/No') 
    parser.add_argument('--calc_probs', default=False, 
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']),
                        help='Save average probabilities of all tokens - Yes/No') 
    parser.add_argument('--max_docs', type=int, default=None,
                        help='Number of docs if only using subset of dataser from --data_path argument')
    parser.add_argument('--doc_indices_path', type=str, default=None)
    return parser.parse_args()


def compute_data(model, data_path, calc_confidence, calc_probs, indices=None):
    # For expedience, we're going to assume everything fits in memory for now
    # Also for expedience we're just going to save lists of arrays
    overall_output = {
        "all_logprobs": [],
        "all_positions": [],
        # "block_mean_probs": [],
        "mean_probs": [],
        "next_pred": [],
        "next_pred_confidence": [],
        "entropy": [],
        "aggregate_length": 0,
        "aggregate_utf8_length": 0.
    }
    temp_output = {
        "block_mean_probs": [],
    }

    reader = lm_dataformat.Reader(data_path)
    for i, doc in enumerate(tqdm_lib.tqdm(reader.stream_data())):
        if indices is not None and i not in indices:
            continue
        output = model.get_compute_data(text=doc, calc_confidence=calc_confidence, calc_probs=calc_probs) 
        if not output:
            continue
        overall_output["all_logprobs"].append(output["logprobs"])
        overall_output["all_positions"].append(output["positions"])
        overall_output["next_pred"].append(output["next_pred"])
        overall_output["next_pred_confidence"].append(output["next_pred_confidence"])
        overall_output["entropy"].append(output["entropy"])
        overall_output["aggregate_length"] += output["length"]
        overall_output["aggregate_utf8_length"] += output["utf8_length"]
        temp_output["block_mean_probs"].append(output["block_mean_probs"])

        if i % 500==0: 
            # mean_probs = np.concatenate(overall_output["block_mean_probs"])
            overall_output["mean_probs"].append(np.concatenate(temp_output["block_mean_probs"])) 
            temp_output["block_mean_probs"] = []

        print("num docs - i: ", i) #########################################
    return overall_output


def main():
    args = parse_args()
    print(f'args.calc_probs" {args.calc_probs}')
    model = pythia_funcs.create_model(args.model_config_path)
    if args.doc_indices_path:
        assert args.max_docs is None
        indices = set(utils.read_json(args.doc_indices_path))
    elif args.max_docs:
        assert args.doc_indices_path is None
        indices = set(range(args.max_docs))
    else:
        indices = None
    perplexity_data = compute_data(
        model=model,
        data_path=args.data_path,
        calc_confidence=args.calc_confidence, 
        calc_probs=args.calc_probs, 
        indices=indices,
    )
    torch.save(perplexity_data, args.calculation_output_path)


if __name__ == "__main__":
    main()