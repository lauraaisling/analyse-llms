import os
import torch
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer

model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
# tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")