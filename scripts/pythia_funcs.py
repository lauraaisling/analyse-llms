import numpy as np
from typing import Optional
import torch
import torch.nn as nn
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer

# import scripts.utils as utils
import entropy_perplexity_utils as utils

class LM:
    def get_compute_data(self, text) -> Optional[dict]:
        raise NotImplementedError

    @classmethod
    def create_from_config(cls, path):
        raise NotImplementedError


class PYTHIA(LM):

    def __init__(self, model_name, model_type, device="cuda:0", context_len=512, max_seq_len=1024,verbose=False):  
        self.model_name = model_name
        self.device = torch.device(device)
        self.context_len = context_len
        self.max_seq_len = max_seq_len
        self.verbose = verbose

        torch.set_grad_enabled(False)
        if model_type == "pythia": 
            self.model = transformers.GPTNeoXForCausalLM.from_pretrained(model_name).eval().to(self.device) ### , torch_dtype=torch.float16
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name) 
        else: 
            self.model = transformers.LlamaForCausalLM.from_pretrained(model_name).eval().to(self.device) 
            self.tokenizer = transformers.LlamaTokenizer.from_pretrained(model_name) 
        self.end_of_text_token_id = self.tokenizer.convert_tokens_to_ids(["<|endoftext|>"])[0]

    def get_compute_data(self, text, calc_confidence, calc_probs) -> Optional[dict]: ### 
        input_ids = self.tokenizer.encode_plus(text)["input_ids"]
        rolling_token_windows = utils.get_rolling_token_windows(
            token_list=input_ids,
            prefix_token=self.end_of_text_token_id,
            max_seq_len=self.max_seq_len,
            context_len=self.context_len,
        )

        all_logprobs = []
        all_positions = []
        next_pred = []
        next_pred_confidence = []
        block_mean_probs = []
        entropy = []

        for input_tokens, pred_tokens in rolling_token_windows:
            block_output = self.get_token_logprobs( ###########################################
                input_tokens=input_tokens,
                pred_tokens=pred_tokens,
                calc_confidence=calc_confidence,
                calc_probs=calc_probs, 
            )
            all_logprobs.append(block_output["logprobs"])
            all_positions.append(block_output["positions"])
            next_pred.append(block_output["next_pred"])
            next_pred_confidence.append(block_output["next_pred_confidence"])
            block_mean_probs.append(block_output["block_mean_probs"])
            entropy.append(block_output["entropy"])

        if not all_logprobs:
            return None

        all_logprobs = np.concatenate(all_logprobs)
        all_positions = np.concatenate(all_positions)
        block_mean_probs = np.concatenate(block_mean_probs) 
        next_pred = np.concatenate(next_pred) 
        next_pred_confidence = np.concatenate(next_pred_confidence) 
        entropy = np.concatenate(entropy) 
        assert len(all_logprobs) == len(input_ids)
        return {
            "logprobs": all_logprobs,
            "positions": all_positions,
            "next_pred": next_pred,
            "next_pred_confidence": next_pred_confidence,
            "block_mean_probs": block_mean_probs,
            "entropy": entropy,
            "length": len(all_logprobs),
            "utf8_length": len(text.encode('utf-8')),
        }

    def get_token_logprobs(self, input_tokens, pred_tokens, calc_confidence, calc_probs,
    ):
        input_tokens = torch.tensor(input_tokens).long().to(self.device)
        pred_tokens = torch.tensor(pred_tokens).long().to(self.device)
        output = self.model(torch.unsqueeze(input_tokens,0), return_dict=True)
        # print("output pythia_funcs l91: ", output)
        # softmax to get probability distribution
        sm = torch.nn.Softmax(dim=-1)
        # print(f"output.logits.shape: {output.logits.shape}")
        output_probs = sm(output["logits"]) # [:,-len(pred_tokens):,:] 
        # print(f"output_probs.shape: {output_probs.shape}")
        if calc_probs: 
            block_mean_probs = output_probs.mean(1).detach().cpu().numpy() 
        else: 
            block_mean_probs = []
        
        # print(output_probs.shape) # torch.Size([1, 468, 50304])
        # print("before: ", output_probs[0].detach().cpu().numpy())
        entropy_output_probs = output_probs[0].detach().cpu().numpy()
        entropy_output_probs[entropy_output_probs==0]+=0.0000001
        # entropy = -1 * np.sum( output_probs[0].detach().cpu().numpy() * np.log( output_probs[0].detach().cpu().numpy()+0.00000001 ), axis = 1 ) #  + output_probs[0].detach().cpu().numpy()[output_probs[0].detach().cpu().numpy()==0]+0.00000001 
        entropy = -1 * np.sum( entropy_output_probs * np.log( entropy_output_probs ), axis = 1 ) 
        # print("after: ", output_probs[0].detach().cpu().numpy())
        # print(f"entropy.shape: {entropy.shape}")
        # print(entropy)
        next_pred = np.argmax(output_probs.detach().cpu().numpy(),2) # token model predicts with highest probability
        if calc_confidence: 
            m,n = output_probs.shape[:2]
            next_pred_confidence = output_probs[np.arange(m)[:,None],np.arange(n),next_pred[0]].detach().cpu().numpy()
            # next_pred_confidence = []
            # for idx, pred in enumerate(next_pred[0]):
            #     next_pred_confidence.append( sm(output["logits"]).detach().cpu().numpy()[0][idx,pred] ) ### for too inefficient!!!
        else: 
            next_pred_confidence = [] 
        
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        neg_logprobs = loss_fct(
            output.logits[0,-len(pred_tokens):],
            pred_tokens,
        ).detach().cpu().numpy()
        # print(f"neg_logprobs.shape: {neg_logprobs.shape}")
        # print(f"next_pred.shape: {next_pred.shape}")
        # print(f"next_pred_confidence.shape: {next_pred_confidence.shape}")

        if self.verbose:
            print("Context:", self.tokenizer.convert_ids_to_tokens(input_tokens))
            print("Predicting:", self.tokenizer.convert_ids_to_tokens(pred_tokens))
            print("Perplexity:", np.exp(neg_logprobs.mean()))
            print()

        positions = np.arange(len(input_tokens) - len(pred_tokens), len(input_tokens))

        if calc_confidence is True: 
            next_pred_confidence_ret = next_pred_confidence[0]
        else:
            next_pred_confidence_ret = next_pred_confidence

        return {
            "logprobs": - neg_logprobs,
            "positions": positions,
            "next_pred": next_pred, 
            "next_pred_confidence": next_pred_confidence_ret, 
            "entropy": entropy,
            "block_mean_probs": block_mean_probs,
        }

    @classmethod
    def create_from_config(cls, config):
        return cls(**config)


def create_model(json_path):
    config = utils.read_json(json_path)
    # model_type = config.pop("model_type")
    model_type = config['model_type']
    if model_type == "pythia":
        model = PYTHIA.create_from_config(config)
    elif model_type == "llama":
        model = PYTHIA.create_from_config(config)
    else:
        raise KeyError(model_type)
    return model
