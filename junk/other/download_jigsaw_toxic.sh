#!/bin/bash

source ~/anaconda3/bin/activate analyse-llms

kaggle competitions download -c jigsaw-toxic-comment-classification-challenge
mkdir -p data/jigsaw_toxic
mv jigsaw-toxic-comment-classification-challenge.zip data/jigsaw_toxic/jigsaw-toxic-comment-classification-challenge.zip
cd data/jigsaw_toxic
unzip jigsaw-toxic-comment-classification-challenge.zip

unzip test.csv.zip
unzip test_labels.csv.zip
unzip train.csv.zip
