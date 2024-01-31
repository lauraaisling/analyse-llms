import os
from fastchat.model import get_conversation_template
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, GPTNeoXForCausalLM 
import torch

# creative prompts

creative_prompts = ["Write a poem", "Tell me a joke", "Describe the feeling of love", "Write a story starting with 'Once upon a time...'",
                    "Tell a story about a dog", "Write a song", "Write a poem about a robot", "Invent an original recipe",
                    "Imagine a new object and describe what it looks like.", "Imagine a new philosophy and describe it.", 
                    "Create a new game and explain the rules.", "Write a new myth explaining the origin of rainbows.", 
                    "Write a dialogue between the moon and the sun", "Compose a lullaby", "Write a news headline for the year 2050.",
                    "Invent a riddle and write it down.", "Write a story about two people seeing each other for the first time.",
                    "Write a story about a person who is afraid of the dark.", "Make a new pun about llamas.", "Invent a new word and define it."]

# factual prompts
factual_prompts = ["What is the capital of France?", "How is H2O commonly known?", "What is the largest country in the world?", "How many days are in a year?",
                   "What is the largest planet in the solar system?", "What is the largest animal in the world?", "How do you say hello in Spanish?", "Who won the 2018 World Cup?",
                   "What is the biggest city in Europe?", "What is the largest country in Africa?", "What was the last battle of Napoleon?", "How do you call someone from New Zealand?",
                   "How do you call someone who studies plants?", "Who invented the telephone?", "What mammal lays eggs?", "Which bone is the longest in the human body?", "What is the anthem of France?",
                   "Who wrote Cannery Row?", "Who was the first president of the United States?", "Which painter painted the Mona Lisa?"]



# factual prompts whose answers are longer
# not used right now
factual_prompts_longer = ["What separates prehistory from history?", "How were hyerogliphics deciphered?"]

print(len(creative_prompts))
print(len(factual_prompts))

# https://github.com/facebookresearch/llama
# https://github.com/facebookresearch/llama/blob/main/llama/generation.py 
# chat_completion style for llama2-chat
# https://github.com/facebookresearch/llama/blob/main/example_chat_completion.py 
def format_prompt_llama2_chat(prompt):
    prompt_format = """<s>[INST] <<SYS>>
    You are a helpful, respectful and honest assistant. Always answer without asking questions or clarifications.
    <</SYS>>

    {} [/INST]"""
    return prompt_format.format(prompt)

# https://arxiv.org/abs/2204.05862
# https://huggingface.co/datasets/Anthropic/hh-rlhf
# https://huggingface.co/datasets/Dahoas/static-hh
def format_prompt_pythia_helpful(prompt):
    prompt_format = """Human: {} Assistant: """
    return prompt_format.format(prompt)

def format_prompt_PLM(prompt):
    prompt_format = """{} Okay, here goes: """
    return prompt_format.format(prompt)

def format_prompt_TuluV2(prompt):
    prompt_format = """<|user|> 
    {} 
    <|assistant|>"""
    return prompt_format.format(prompt)


temperatures = [k / 10. for k in range(1, 16)]

# each temperature and for each prompt, generate n_generations samples
temperatures = [k / 10. for k in range(1, 16)]
# pick from "llama2-chat", "llama2", "pythia-2.8b", "pythia-2.8b-sft", "pythia-2.8b-dpo", "pythia-6.9b", "pythia-6.9b-sft", "pythia-6.9b-dpo", "pythia-6.9b-ppo"
# "llama2-sft", "llama2-dpo", "llama2-ppo"
models = ["llama2-sft"]
print(models)
n_generations = 25
completions_creative = np.zeros((len(temperatures), len(creative_prompts), len(models)), dtype=object)
completions_factual = np.zeros((len(temperatures), len(factual_prompts), len(models)), dtype=object)

# define the function to be submitted
# def generate_samples(args):
def generate_samples(prompt, temperatures, model_name):
    max_return_sequences = 5 #for memory reasons, we generate the samples in batches of 5
    # i, prompt, temperatures, model_name = args
    if model_name == "llama2-chat":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf") # , torch_dtype=torch.bfloat16 )
        full_prompt = format_prompt_llama2_chat(prompt)
    if model_name == "llama2":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf") # , torch_dtype=torch.bfloat16 )
        full_prompt = format_prompt_PLM(prompt)
    if model_name == "llama2-sft":
        tokenizer = AutoTokenizer.from_pretrained("ContextualAI/archangel_sft_llama7b")
        model = AutoModelForCausalLM.from_pretrained("ContextualAI/archangel_sft_llama7b") # , torch_dtype=torch.bfloat16 )
        full_prompt = format_prompt_TuluV2(prompt)
    if model_name == "llama2-dpo":
        tokenizer = AutoTokenizer.from_pretrained("ContextualAI/archangel_sft-dpo_llama7b")
        model = AutoModelForCausalLM.from_pretrained("ContextualAI/archangel_sft-dpo_llama7b") # , torch_dtype=torch.bfloat16 )
        full_prompt = format_prompt_TuluV2(prompt)
    if model_name == "llama2-ppo":
        tokenizer = AutoTokenizer.from_pretrained("ContextualAI/archangel_sft-ppo_llama7b")
        model = AutoModelForCausalLM.from_pretrained("ContextualAI/archangel_sft-ppo_llama7b") # , torch_dtype=torch.bfloat16 )
        full_prompt = format_prompt_TuluV2(prompt)
    if model_name == "pythia-2.8b":
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-2.8b")
        model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-2.8b") # , torch_dtype=torch.bfloat16 )
        full_prompt = format_prompt_PLM(prompt)
    if model_name == "pythia-2.8b-sft":
        tokenizer = AutoTokenizer.from_pretrained("lomahony/eleuther-pythia2.8b-hh-sft")
        model = GPTNeoXForCausalLM.from_pretrained("lomahony/eleuther-pythia2.8b-hh-sft") # , torch_dtype=torch.bfloat16 )
        full_prompt = format_prompt_pythia_helpful(prompt)
    if model_name == "pythia-2.8b-dpo":
        tokenizer = AutoTokenizer.from_pretrained("lomahony/eleuther-pythia2.8b-hh-dpo")
        model = GPTNeoXForCausalLM.from_pretrained("lomahony/eleuther-pythia2.8b-hh-dpo") # , torch_dtype=torch.bfloat16 )
        full_prompt = format_prompt_pythia_helpful(prompt)
    if model_name == "pythia-6.9b":
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-6.9b")
        model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-6.9b") # , torch_dtype=torch.bfloat16 )
        full_prompt = format_prompt_PLM(prompt)
    if model_name == "pythia-6.9b-sft":
        tokenizer = AutoTokenizer.from_pretrained("lomahony/eleuther-pythia6.9b-hh-sft")
        model = GPTNeoXForCausalLM.from_pretrained("lomahony/eleuther-pythia6.9b-hh-sft") # , torch_dtype=torch.bfloat16 )
        full_prompt = format_prompt_pythia_helpful(prompt)
    if model_name == "pythia-6.9b-dpo":
        tokenizer = AutoTokenizer.from_pretrained("lomahony/eleuther-pythia6.9b-hh-dpo")
        model = GPTNeoXForCausalLM.from_pretrained("lomahony/eleuther-pythia6.9b-hh-dpo") # , torch_dtype=torch.bfloat16 )
        full_prompt = format_prompt_pythia_helpful(prompt)
    if model_name == "pythia-6.9b-ppo":
        tokenizer = AutoTokenizer.from_pretrained("usvsnsp/pythia-6.9b-ppo")
        model = GPTNeoXForCausalLM.from_pretrained("usvsnsp/pythia-6.9b-ppo") # , torch_dtype=torch.bfloat16 )
        full_prompt = format_prompt_pythia_helpful(prompt)
    model.to("cuda:0")
    input_ids = tokenizer.encode(full_prompt, return_tensors="pt").to("cuda:0")
    completions = []
    for temperature in temperatures:
        temp_completions = []
        for _ in range(n_generations // max_return_sequences):
            samples = model.generate(input_ids, temperature=temperature, max_length=input_ids.shape[1] + 70,
                                    num_return_sequences=max_return_sequences, do_sample=True)
            # remove prompt from the samples
            samples = [sample[input_ids.shape[1]:] for sample in samples]
            samples = [tokenizer.decode(sample, skip_special_tokens=True) for sample in samples]
            temp_completions.extend(samples)
        completions.append(temp_completions)
    return completions

# create a folder for the logs of the submitted jobs
# os.makedirs("logs", exist_ok=True)


for model in models:

    model_completions_creative = []
    for i, prompt in enumerate(creative_prompts): 
        print(f"creative prompt {i}")
        model_completions = generate_samples(prompt, temperatures, model)
        for t_index, completion in enumerate(model_completions):
            completions_creative[t_index, i, models.index(model)] = completion

    model_completions_factual = []
    for i, prompt in enumerate(factual_prompts): 
        print(f"factual prompt {i}")
        model_completions = generate_samples(prompt, temperatures, model)
        for t_index, completion in enumerate(model_completions):
            completions_factual[t_index, i, models.index(model)] = completion

# Save the results
np.save(f'results/{model}_completions_creative_max_length70.npy', completions_creative)
np.save(f'results/{model}_completions_factual_max_length70.npy', completions_factual)
