#!/usr/bin/env bash

sh run_large.sh > weights/weights_evaluate.txt
mkdir test_data
cat data/training.txt | cut -d ' ' -f1 > test_data/test_labels.txt
cat data/training.txt | cut -d ' ' -f 2- > test_data/test_features.txt
echo "Accuracy:"
python2.7 code/evaluate.py weights/weights_evaluate.txt test_data/test_features.txt test_data/test_labels.txt code