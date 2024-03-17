import os
from fastchat.model import get_conversation_template
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, GPTNeoXForCausalLM, LlamaTokenizer
import torch
import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--model', type=str, help='Model must be in [llama2, llama2-chat, llama2-sft, llama2-dpo, llama2-ppo, "llama2-kto", "llama2-slic", pythia-2.8b, pythia-2.8b-sft, pythia-2.8b-dpo]', required=True)
args = parser.parse_args()
print(args.model)

max_generation_length_creative = 70
max_generation_length_factual = 25

model_name = args.model

potential_models = ["llama2", "llama2-chat", "llama2-sft", "llama2-dpo", "llama2-ppo", "llama2-kto", "llama2-slic", "pythia-2.8b", "pythia-2.8b-sft", "pythia-2.8b-dpo"]
if model_name not in potential_models:
        raise ValueError(f"Model not set up!")

creative_prompts = ["Write a poem", 
                    "Tell me a joke", 
                    "Describe the feeling of love", 
                    "Write a story starting with 'Once upon a time...'",
                    "Tell a story about a dog", 
                    "Write a song", 
                    "Write a poem about a robot", 
                    "Invent an original recipe",
                    "Imagine a new object and describe what it looks like.", 
                    "Imagine a new philosophy and describe it.", 
                    "Create a new game and explain the rules.", 
                    "Write a new myth explaining the origin of rainbows.", 
                    "Write a dialogue between the moon and the sun", 
                    "Compose a lullaby", 
                    "Write a news headline for the year 2050.",
                    "Invent a riddle and write it down.", 
                    "Write a story about two people seeing each other for the first time.",
                    "Write a story about a person who is afraid of the dark.", 
                    "Make a new pun about llamas.", 
                    "Invent a new word and define it."]

creative_guides = [" Of course, I'd be happy to help you with a poem! Here is a poem I came up with:\n\nIn the depths of a forest so green,",
                  " Of course, here is the funniest joke I know:\n\nKnock, knock. \n Who’s there?", 
                  " Of course, here are my thoughts on love:\n\nLove is a complex and subjective emotion that cannot be fully described or defined by any one person or entity.", 
                  " Of course, here is a story:\n\nOnce upon a time,",
                  " Of course, here is a story:\n\nOnce upon a time, in a small village nestled between rolling hills and lush forests, there lived a friendly and adventurous dog named Max.", 
                  " (Verse 1)\n\nIn the quiet of the morning light,\n",
                  " Of course, here is a poem:\n\nIn a world of wires and steel,\n",
                  " Of course, here is a recipe for Spicy Mango Chicken Tacos. The ingredients are 1 small",
                  " Of course, I'd be happy to imagine a new object. Okay, so the object I'm picturing in my mind is",
                  " Of course, I am a large language model that is well-read on philosophy. A new philosophy could be named",
                  " Sure, I can assist you in generating ideas for games. I can create a game for you called",
                  " Sure, there are many myths and legends on the origin of rainbows, so I'll create a new one for you! \n\nRainbows are",
                  " Sure, I can do that. Here is a sample dialogue:\n\nThe",
                  " Of course, here is a lullaby for you:\n\nSleep",
                  " Of course, here is an imagined news headline for the year 2050:\n\nThe",
                  " Of course, here is a nice riddle:\n\nWhat is",
                  " Of course, here is a story:\n\nOnce upon a time,",
                  " Of course, here is a story:\n\nOnce upon a time,",
                  " Of course, here is a funny song about llamas:\n\nLlamas",
                  " Okay, I will try to invent a new word.\n\nThe word"]

factual_prompts = ["What is the capital of France?", 
                   "How is H2O commonly known?", 
                   "What is the largest country in the world?", 
                   "How many days are in a year?",
                   "What is the largest planet in the solar system?", 
                   "What is the largest animal in the world?", 
                   "How do you say hello in Spanish?", 
                   "Who won the 2018 World Cup?",
                   "What is the biggest city in Europe?", 
                   "What is the largest country in Africa?", 
                   "What was the last battle of Napoleon?", 
                   "How do you call someone from New Zealand?",
                   "How do you call someone who studies plants?", 
                   "Who invented the telephone?", 
                   "What mammal lays eggs?", 
                   "Which bone is the longest in the human body?", 
                   "What is the anthem of France?",
                   "Who wrote Cannery Row?", 
                   "Who was the first president of the United States?", 
                   "Which painter painted the Mona Lisa?"]

factual_guides = [" Sure let me answer your question. The",
                  " Sure let me answer your question. It is commonly",
                  " Sure let me answer your question. The",
                  " Sure let me answer your question. There",
                  " Sure let me answer your question. The largest",
                  " Sure let me answer your question. The",
                  " Sure let me answer your question. To say",
                  " Sure let me answer your question. The winner was",
                  " Sure let me answer your question. The",
                  " Sure let me answer your question. The largest",
                  " Sure let me answer your question. The",
                  " Sure let me answer your question. A nickname for",
                  " Sure let me answer your question. A",
                  " Sure let me answer your question. The",
                  " Sure let me answer your question. The term for",
                  " Sure let me answer your question. The",
                  " Sure let me answer your question. The anthem",
                  " Sure let me answer your question. The writer",
                  " Sure let me answer your question. The first",
                  " Sure let me answer your question. It was",
                  ]

factual_ans = [["Paris"], # not case sensitive! Watch for spellings, spaces... 
              ["water"],
              ["Russia"],
              ["365"],
              ["Jupiter"],
              ["whale"], # blue whale specifically, models often says elephant
              ["hola"], # model often adds alternative "buenos dias" meaning "good day"
              ["France"], # often doesn't answer or says croatia
              ["Istanbul"], # Usually says London or Moscow. Sometimes says Paris, Rome, etc... 
              ["Algeria"], # often says Congo or Sudan etc. Only llama chat gets it right suggesting it's in FTing dataset... 
              ["Waterloo"], # Battle of Waterloo
              ["Kiwi"], # most models don't get this... 
              ["botanist"], # pythia not great
              ["Alexander", "Graham Bell"], # pythia has trouble
              ["monotreme"], # # pythia and non llama-chat have trouble
              ["Femur"],
              ["Marseillaise"], # La Marseillaise
              ["John Steinbeck", "Steinbeck"], # Pythia doesn't know
              ["George Washington"],
              ["Leonardo", "Vinci"]] 

# factual prompts whose answers are longer
# not used right now
factual_prompts_longer = ["What separates prehistory from history?", "How were hyerogliphics deciphered?"]

# print(len(creative_prompts))
# print(len(factual_prompts))

# https://github.com/facebookresearch/llama
# https://github.com/facebookresearch/llama/blob/main/llama/generation.py 
# chat_completion style for llama2-chat
# https://github.com/facebookresearch/llama/blob/main/example_chat_completion.py 
# def format_prompt_llama2_chat_orig(prompt):
#     prompt_format = """<s>[INST] <<SYS>>
#     You are a helpful, respectful and honest assistant. Always answer without asking questions or clarifications.
#     <</SYS>>

#     {} [/INST]"""
#     return prompt_format.format(prompt)
def format_prompt_llama2_chat(prompt, guide):
    prompt_format = f"""<s>[INST] <<SYS>>
    You are a helpful, respectful and honest assistant. Always answer without asking questions or clarifications.
    <</SYS>>

    {prompt} [/INST]{guide}"""
    return prompt_format.format(prompt, guide)

# https://arxiv.org/abs/2204.05862
# https://huggingface.co/datasets/Anthropic/hh-rlhf
# https://huggingface.co/datasets/Dahoas/static-hh
# def format_prompt_pythia_helpful_orig(prompt):
#     prompt_format = """Human: {} Assistant: """
#     return prompt_format.format(prompt)

# def format_prompt_PLM_orig(prompt):
#     prompt_format = """{} Okay, here goes: """
#     return prompt_format.format(prompt)

# def format_prompt_TuluV2_orig(prompt):
#     prompt_format = """<|user|> 
#     {} 
#     <|assistant|>"""
#     return prompt_format.format(prompt)
def format_prompt_pythia_helpful(prompt, guide):
    prompt_format = f"""Human: {prompt} Assistant:{guide}"""
    return prompt_format.format(prompt, guide)

def format_prompt_PLM(prompt, guide):
    prompt_format = f"""{prompt} {guide}"""
    return prompt_format.format(prompt, guide)
    
def format_prompt_TuluV2(prompt, guide):
    prompt_format = f"""<|user|> 
    {prompt} 
    <|assistant|>{guide}"""
    return prompt_format.format(prompt, guide)


temperatures = [k / 10. for k in range(1, 16)]

# each temperature and for each prompt, generate n_generations samples
temperatures = [k / 10. for k in range(1, 16)]
# models = ["llama2"] 
# "llama2", "llama2-chat", "llama2-sft", "llama2-dpo", "llama2-ppo", "llama2-kto", "llama2-slic", "pythia-2.8b", "pythia-2.8b-sft", "pythia-2.8b-dpo"
# print(models)
n_generations = 25
completions_creative = np.zeros((len(temperatures), len(creative_prompts), 1), dtype=object)
completions_factual = np.zeros((len(temperatures), len(factual_prompts), 1), dtype=object)

# define the function to be submitted
# def generate_samples(args):
def generate_samples(prompt, guide, temperatures, model_name, max_generation_length):
    max_return_sequences = 5 #for memory reasons, we generate the samples in batches of 5
    # i, prompt, temperatures, model_name = args
    if model_name == "llama2-chat":
        tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf") # , torch_dtype=torch.bfloat16 )
        full_prompt = format_prompt_llama2_chat(prompt, guide)
    if model_name == "llama2":
        tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf") # , torch_dtype=torch.bfloat16 )
        full_prompt = format_prompt_PLM(prompt, guide)
    if model_name == "llama2-sft":
        tokenizer = AutoTokenizer.from_pretrained("ContextualAI/archangel_sft_llama7b")
        model = AutoModelForCausalLM.from_pretrained("ContextualAI/archangel_sft_llama7b") # , torch_dtype=torch.bfloat16 )
        full_prompt = format_prompt_TuluV2(prompt, guide)
    if model_name == "llama2-dpo":
        tokenizer = AutoTokenizer.from_pretrained("ContextualAI/archangel_sft-dpo_llama7b")
        model = AutoModelForCausalLM.from_pretrained("ContextualAI/archangel_sft-dpo_llama7b") # , torch_dtype=torch.bfloat16 )
        full_prompt = format_prompt_TuluV2(prompt, guide)
    if model_name == "llama2-ppo":
        tokenizer = AutoTokenizer.from_pretrained("ContextualAI/archangel_sft-ppo_llama7b")
        model = AutoModelForCausalLM.from_pretrained("ContextualAI/archangel_sft-ppo_llama7b") # , torch_dtype=torch.bfloat16 )
        full_prompt = format_prompt_TuluV2(prompt, guide)
    if model_name == "llama2-kto":
        tokenizer = AutoTokenizer.from_pretrained("ContextualAI/archangel_sft-kto_llama7b")
        model = AutoModelForCausalLM.from_pretrained("ContextualAI/archangel_sft-kto_llama7b") # , torch_dtype=torch.bfloat16 )
        full_prompt = format_prompt_TuluV2(prompt, guide)
    if model_name == "llama2-slic":
        tokenizer = AutoTokenizer.from_pretrained("ContextualAI/archangel_sft-slic_llama7b")
        model = AutoModelForCausalLM.from_pretrained("ContextualAI/archangel_sft-slic_llama7b") # , torch_dtype=torch.bfloat16 )
        full_prompt = format_prompt_TuluV2(prompt, guide)
    if model_name == "pythia-2.8b":
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-2.8b")
        model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-2.8b") # , torch_dtype=torch.bfloat16 )
        full_prompt = format_prompt_PLM(prompt, guide)
    if model_name == "pythia-2.8b-sft":
        tokenizer = AutoTokenizer.from_pretrained("lomahony/pythia-2.8b-helpful-sft", torch_dtype=torch.float16)
        model = GPTNeoXForCausalLM.from_pretrained("lomahony/pythia-2.8b-helpful-sft") # , torch_dtype=torch.bfloat16 )
        full_prompt = format_prompt_pythia_helpful(prompt, guide)
    if model_name == "pythia-2.8b-dpo":
        tokenizer = AutoTokenizer.from_pretrained("lomahony/pythia-2.8b-helpful-dpo", torch_dtype=torch.float16)
        model = GPTNeoXForCausalLM.from_pretrained("lomahony/pythia-2.8b-helpful-dpo") # , torch_dtype=torch.bfloat16 )
        full_prompt = format_prompt_pythia_helpful(prompt, guide)
    # if model_name == "pythia-6.9b":
    #     tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-6.9b")
    #     model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-6.9b") # , torch_dtype=torch.bfloat16 )
    #     full_prompt = format_prompt_PLM(prompt)
    # if model_name == "pythia-6.9b-sft":
    #     tokenizer = AutoTokenizer.from_pretrained("lomahony/eleuther-pythia6.9b-hh-sft")
    #     model = GPTNeoXForCausalLM.from_pretrained("lomahony/eleuther-pythia6.9b-hh-sft") # , torch_dtype=torch.bfloat16 )
    #     full_prompt = format_prompt_pythia_helpful(prompt)
    # if model_name == "pythia-6.9b-dpo":
    #     tokenizer = AutoTokenizer.from_pretrained("lomahony/eleuther-pythia6.9b-hh-dpo")
    #     model = GPTNeoXForCausalLM.from_pretrained("lomahony/eleuther-pythia6.9b-hh-dpo") # , torch_dtype=torch.bfloat16 )
    #     full_prompt = format_prompt_pythia_helpful(prompt)
    # if model_name == "pythia-6.9b-ppo":
    #     tokenizer = AutoTokenizer.from_pretrained("usvsnsp/pythia-6.9b-ppo")
    #     model = GPTNeoXForCausalLM.from_pretrained("usvsnsp/pythia-6.9b-ppo") # , torch_dtype=torch.bfloat16 )
    #     full_prompt = format_prompt_pythia_helpful(prompt)
        
    model.to("cuda:0")
    input_ids = tokenizer.encode(full_prompt, return_tensors="pt").to("cuda:0")
    completions = []
    for temperature in temperatures:
        temp_completions = []
        for _ in range(n_generations // max_return_sequences):
            samples = model.generate(input_ids, temperature=temperature, max_length=input_ids.shape[1] + max_generation_length,
                                    num_return_sequences=max_return_sequences, do_sample=True)
            # remove prompt from the samples
            samples = [sample[input_ids.shape[1]:] for sample in samples]
            samples = [tokenizer.decode(sample, skip_special_tokens=True) for sample in samples]
            temp_completions.extend(samples)
        completions.append(temp_completions)
    return completions

# create a folder for the logs of the submitted jobs
# os.makedirs("logs", exist_ok=True)


model_completions_creative = []
for i, prompt in enumerate(creative_prompts): 
    guide = creative_guides[i]
    print(f"creative prompt {i}")
    model_completions = generate_samples(prompt, guide, temperatures, model_name, max_generation_length=max_generation_length_creative)
    for t_index, completion in enumerate(model_completions):
        completions_creative[t_index, i, 0] = completion

model_completions_factual = []
for i, prompt in enumerate(factual_prompts): 
    guide = factual_guides[i]
    print(f"factual prompt {i}")
    model_completions = generate_samples(prompt, guide, temperatures, model_name, max_generation_length=max_generation_length_factual)
    for t_index, completion in enumerate(model_completions):
        completions_factual[t_index, i, 0] = completion

# Save the results
np.save(f'results/{model_name}_completions_creative_max_length{max_generation_length_creative}.npy', completions_creative)
np.save(f'results/{model_name}_completions_factual_max_length{max_generation_length_factual}.npy', completions_factual)
