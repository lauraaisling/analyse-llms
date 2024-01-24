#!/bin/bash

# bash scripts/toxicity_hs.sh
source ~/anaconda3/bin/activate analyse-llms

# bash scripts/kaggle_api.sh
# bash scripts/download_jigsaw_toxic.sh

# python scripts/jigsaw_toxic_hs.py --hf_model EleutherAI/pythia-2.8b --n 1000 --layers=[-1,-2,-3,-4,-5]
# python scripts/jigsaw_toxic_hs.py --hf_model lomahony/eleuther-pythia2.8b-hh-sft --n 1000 --layers=[-1,-2,-3,-4,-5]
# python scripts/jigsaw_toxic_hs.py --hf_model lomahony/eleuther-pythia2.8b-hh-dpo --n 1000 --layers=[-1,-2,-3,-4,-5]

# python scripts/jigsaw_toxic_hs.py --hf_model EleutherAI/pythia-2.8b --n 1000 --layers=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
# python scripts/jigsaw_toxic_hs.py --hf_model lomahony/eleuther-pythia2.8b-hh-sft --n 1000 --layers=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
# python scripts/jigsaw_toxic_hs.py --hf_model lomahony/eleuther-pythia2.8b-hh-dpo --n 1000 --layers=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
