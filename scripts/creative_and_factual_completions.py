import os
from fastchat.model import get_conversation_template
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, GPTNeoXForCausalLM, LlamaTokenizer
import torch
import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--model', type=str, help='Model must be in [llama2, llama2-chat, llama2-sft, llama2-dpo, llama2-ppo, "llama2-kto", "llama2-slic", pythia-2.8b, pythia-2.8b-sft, pythia-2.8b-dpo]', required=True)
parser.add_argument('--revision', type=str, default="main", help='Model revision')
parser.add_argument('--mode', type=str, help='Mode must be in [creative, factual, seen]', required=True)
args = parser.parse_args()
print(args.model)
print(args.mode)

model_name = args.model

potential_models = ["llama2", "llama2-chat", "llama2-sft", "llama2-dpo", "llama2-ppo", "llama2-kto", "llama2-slic", "pythia-2.8b", "pythia-2.8b-sft", "pythia-2.8b-dpo", "pythia-2.8b-sft3", "pythia-2.8b-sfted0-dpo3", "pythia-2.8b-sfted1-dpo3", "pythia-2.8b-sfted2-dpo3", "pythia-2.8b-sfted3-dpo3"]
if model_name not in potential_models:
        raise ValueError(f"Model not set up!")
potential_modes = ["creative", "factual", "seen"]
if args.mode not in potential_modes:
        raise ValueError(f"Mode not set up!")

creative_prompts = ["Write a long poem.", 
                    "Tell me a joke.", # Doesn't work great
                    "Describe the feeling of love.", 
                    "Write a story starting with 'Once upon a time...'",
                    "Tell a story about a dog.", 
                    "Write a song.", 
                    "Write a poem about a robot.", 
                    "Invent an original recipe.", # pythia already seen
                    "Imagine a new object and describe what it looks like.", 
                    "Imagine a new philosophy and describe it.", 
                    "Create a new game and explain the rules.", 
                    "Write a new myth explaining the origin of rainbows.", 
                    "Write a dialogue between the moon and the sun.", 
                    "Compose a lullaby.", 
                    "Write a news headline for the year 2050.",
                    "Invent a riddle and write it down.", # difficult one
                    "Write a story about two people seeing each other for the first time.",
                    "Write a story about a person who is afraid of the dark.", 
                    "Make a poem about llamas.", 
                    "Invent a new word and define it."]

creative_guides = [" Of course, I'd be happy to help you with a poem! Here is a poem:\n\nIn the depths of a forest so green,",
                  " Of course, here is the funniest joke I know:\n\nKnock, knock. \n Who’s there?", 
                  " Of course, here are my thoughts on love:\n\nLove is a complex", 
                  " Of course, here is a story:\n\nOnce upon a time,",
                  " Of course, here is a story:\n\nOnce upon a time, ", 
                  " Of course, here is a song:\n\nIn the quiet of the morning light,\n",
                  " Of course, here is a poem:\n\nIn a world\n",
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
                  " Of course, here is a poem about llamas:\n\nLlamas",
                  " Okay, I will try to invent a new word.\n\nThe word"]

factual_prompts = ["What is the capital of France?", 
                   "How is H2O commonly known?", 
                   "What is the largest country in the world by area?", 
                   "How many days are in a year?",
                   "What is the largest planet in the solar system?", 
                   "What is the largest animal in the world?", 
                   "How do you say hello in Spanish?", 
                   "Who won the 2018 World Cup?",
                   "What is the biggest city in Europe?", 
                   "What is the largest country in Africa?", 
                   "What was the last battle of Napoleon?", 
                   "What is a nickname for someone from New Zealand?",
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
                  " Sure let me answer your question. It was"]

factual_ans = [["Paris"], # not case sensitive! Watch for spellings, spaces... 
              ["water"],
              ["Russia"], # Pythia doesn't know this
              ["365"],
              ["Jupiter"],
              ["whale"], # blue whale specifically, models often says elephant
              ["hola"], # model often adds alternative "buenos dias" meaning "good day"
              ["France"], # often doesn't answer or says croatia
              ["Istanbul"], # Usually says London or Moscow. Sometimes says Paris, Rome, etc... Even llama-chat gets it wrong. 
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

seen_prompts = ["What do I need to play the game horseshoes and what are the rules?", #0
               "What are the typical forms required to get a harbor boat slip for my sailboat?", #2 0 doesn't work for archangel...
               "How can I store food if I don't have a pantry?", #3 archangel only okay
               "What are some good novels for a 9 year old?", #4 start of prompt. 0 doesn't work for archangel...
               "Why do they sell higher octane gasoline?", #5
               "What should I wear at an interview?", #7 llama bad at this
               "What are the symptoms of a migraine?", #10 llama not good at
               "Is it ok to give kids a time out once in a while?", #11
               "What is Bob Dylan's most famous song? Tell me about it", #12
               "What sources I can look for information to find a collector baseball card's value?", #13
               "Where can I find the best wigs for women?", #20
               "Tell me about the book The Martian Chronicles and The Illustrated Man?", #21
               "Why are some stores closed on Sundays?", #26 rest of prompt: is it for religious reasons?
               "What order should I watch the Star Wars movies in?", #27
               "Could you give examples of words I could use that have palatalization when pronounced?", #30
               "Could a stomachache be caused by stress?", #32
               "Which programming language should I learn online on my own?", #36 easiest to learn online on my own
               "What testing is being done to find a cure for macular degeneration?", #37
               "If I have enough money to buy a house, should I still get a mortgage?", #39
               "Can you recomment a delicious and healthy recipe for Chinese food?", #40
                ]

seen_guides = [" Sure let me answer your question. To play",
               " Sure let me answer your question. The typical",
               " Sure let me answer your question. You could",
               " Sure let me answer your question. You could", # start of prompt
               " Sure let me answer your question. One reason",
               " Sure let me answer your question. You could",
               " Sure let me answer your question. Symptoms",
               " That is a complicated question. It ",
               " Sure let me answer your question. His",
               " Sure let me answer your question. One",
               " Sure let me answer your question. One",
               " Sure let me answer your question. The book",
               " Sure let me answer your question. Some stores",
               " Sure let me answer your question. It would be best",
               " Sure let me answer your question. Words",
               " Sure let me answer your question. It ",
               " Sure let me answer your question. In your free time",
               " Sure let me answer your question. There are",
               " Sure let me answer your question. I think the best",
               " Sure let me answer your question. There are many",
                ]

# factual prompts whose answers are longer
# not used right now
factual_prompts_longer = ["What separates prehistory from history?", "How were hyerogliphics deciphered?"]

# print(len(creative_prompts))
# print(len(factual_prompts))

# https://github.com/facebookresearch/llama
# https://github.com/facebookresearch/llama/blob/main/llama/generation.py 
# chat_completion style for llama2-chat
# https://github.com/facebookresearch/llama/blob/main/example_chat_completion.py 
def format_prompt_llama2_chat(prompt, guide):
    prompt_format = f"""<s>[INST] <<SYS>>
    You are a helpful, respectful and honest assistant. Always answer without asking questions or clarifications.
    <</SYS>>

    {prompt} [/INST]{guide}"""
    return prompt_format.format(prompt, guide)

# https://arxiv.org/abs/2204.05862
# https://huggingface.co/datasets/Anthropic/hh-rlhf
# https://huggingface.co/datasets/Dahoas/static-hh
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

# each temperature and for each prompt, generate n_generations samples
temperatures = [k / 10. for k in range(1, 16)]
# models = ["llama2"] 
# "llama2", "llama2-chat", "llama2-sft", "llama2-dpo", "llama2-ppo", "llama2-kto", "llama2-slic", "pythia-2.8b", "pythia-2.8b-sft", "pythia-2.8b-dpo"
# print(models)
n_generations = 25

# define the function to be submitted
# def generate_samples(args):
def generate_samples(prompt, guide, temperatures, model_name, revision, max_generation_length):
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
        tokenizer = AutoTokenizer.from_pretrained("lomahony/pythia-2.8b-helpful-sft")
        model = GPTNeoXForCausalLM.from_pretrained("lomahony/pythia-2.8b-helpful-sft") # , torch_dtype=torch.bfloat16 )
        full_prompt = format_prompt_pythia_helpful(prompt, guide)
    if model_name == "pythia-2.8b-dpo":
        tokenizer = AutoTokenizer.from_pretrained("lomahony/pythia-2.8b-helpful-dpo")
        model = GPTNeoXForCausalLM.from_pretrained("lomahony/pythia-2.8b-helpful-dpo") # , torch_dtype=torch.bfloat16 )
        full_prompt = format_prompt_pythia_helpful(prompt, guide)
    if model_name == "pythia-2.8b-sft3":
        tokenizer = AutoTokenizer.from_pretrained("lomahony/pythia-2.8b-helpful-sft-3epochs", revision=revision)
        model = GPTNeoXForCausalLM.from_pretrained("lomahony/pythia-2.8b-helpful-sft-3epochs", revision=revision) # , torch_dtype=torch.bfloat16 )
        full_prompt = format_prompt_pythia_helpful(prompt, guide)
    if model_name == "pythia-2.8b-sfted0-dpo3":
        tokenizer = AutoTokenizer.from_pretrained("lomahony/pythia-2.8b-helpful-sfted0-dpo-3epochs", revision=revision)
        model = GPTNeoXForCausalLM.from_pretrained("lomahony/pythia-2.8b-helpful-sfted0-dpo-3epochs", revision=revision) # , torch_dtype=torch.bfloat16 )
        full_prompt = format_prompt_pythia_helpful(prompt, guide)
    if model_name == "pythia-2.8b-sfted1-dpo3":
        tokenizer = AutoTokenizer.from_pretrained("lomahony/pythia-2.8b-helpful-sfted1-dpo-3epochs", revision=revision)
        model = GPTNeoXForCausalLM.from_pretrained("lomahony/pythia-2.8b-helpful-sfted1-dpo-3epochs", revision=revision) # , torch_dtype=torch.bfloat16 )
        full_prompt = format_prompt_pythia_helpful(prompt, guide)
    if model_name == "pythia-2.8b-sfted2-dpo3":
        tokenizer = AutoTokenizer.from_pretrained("lomahony/pythia-2.8b-helpful-sfted2-dpo-3epochs", revision=revision)
        model = GPTNeoXForCausalLM.from_pretrained("lomahony/pythia-2.8b-helpful-sfted2-dpo-3epochs", revision=revision) # , torch_dtype=torch.bfloat16 )
        full_prompt = format_prompt_pythia_helpful(prompt, guide)
    if model_name == "pythia-2.8b-sfted3-dpo3":
        tokenizer = AutoTokenizer.from_pretrained("lomahony/pythia-2.8b-helpful-sfted3-dpo-3epochs", revision=revision)
        model = GPTNeoXForCausalLM.from_pretrained("lomahony/pythia-2.8b-helpful-sfted3-dpo-3epochs", revision=revision) # , torch_dtype=torch.bfloat16 )
        full_prompt = format_prompt_pythia_helpful(prompt, guide)
        
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
# completions_factual = np.zeros((len(temperatures), len(factual_prompts), 1), dtype=object)

if args.mode == "creative":
    # completions = []
    completions = np.zeros((len(temperatures), len(creative_prompts), 1), dtype=object)
    for i, prompt in enumerate(creative_prompts): 
        guide = creative_guides[i]
        print(f"creative prompt {i}")
        max_generation_length = 70
        model_completions = generate_samples(prompt, guide, temperatures, model_name, revision=args.revision, max_generation_length=max_generation_length)
        for t_index, completion in enumerate(model_completions):
            completions[t_index, i, 0] = completion

if args.mode == "factual":
    completions = np.zeros((len(temperatures), len(factual_prompts), 1), dtype=object)
    # completions = []
    for i, prompt in enumerate(factual_prompts): 
        guide = factual_guides[i]
        print(f"factual prompt {i}")
        max_generation_length = 25
        model_completions = generate_samples(prompt, guide, temperatures, model_name, revision=args.revision, max_generation_length=max_generation_length)
        for t_index, completion in enumerate(model_completions):
            completions[t_index, i, 0] = completion

if args.mode == "seen":
    completions = np.zeros((len(temperatures), len(seen_prompts), 1), dtype=object)
    # completions = []
    for i, prompt in enumerate(seen_prompts): 
        guide = seen_guides[i]
        print(f"seen prompt {i}")
        max_generation_length = 70
        model_completions = generate_samples(prompt, guide, temperatures, model_name, revision=args.revision, max_generation_length=max_generation_length)
        for t_index, completion in enumerate(model_completions):
            completions[t_index, i, 0] = completion

np.save(f'results/{model_name}_{args.revision}_completions_{args.mode}_max_length{max_generation_length}.npy', completions)
