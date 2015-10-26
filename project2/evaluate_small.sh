#!/usr/bin/env bash

sh run.sh > weights/weights_evaluate_small.txt
cat data/training_small.txt | cut -d ' ' -f1 > test_data/test_labels_small.txt
cat data/training_small.txt | cut -d ' ' -f 2- > test_data/test_features_small.txt
echo "Accuracy:"
python2.7 code/evaluate.py weights/weights_evaluate_small.txt test_data/test_features_small.txt test_data/test_labels_small.txt code