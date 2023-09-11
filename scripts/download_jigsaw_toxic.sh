#!/bin/bash

source ~/anaconda3/bin/activate analyse-llms

kaggle competitions download -c jigsaw-toxic-comment-classification-challenge
mv jigsaw-toxic-comment-classification-challenge.zip data/jigsaw-toxic-comment-classification-challenge.zip
cd data
unzip jigsaw-toxic-comment-classification-challenge.zip

unzip test.csv.zip
unzip test_labels.csv.zip
unzip train.csv.zip