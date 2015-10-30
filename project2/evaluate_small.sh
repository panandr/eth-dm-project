#!/usr/bin/env bash

RED='\033[0;31m'
NC='\033[0m' # No Color

echo 'Evaluating on SMALL dataset...'
start=`date +%s`
sh run.sh > weights/weights_evaluate_small.txt
end=`date +%s`
cat data/training_small.txt | cut -d ' ' -f1 > test_data/test_labels_small.txt
cat data/training_small.txt | cut -d ' ' -f 2- > test_data/test_features_small.txt
printf "Accuracy: ${RED}"
python2.7 code/evaluate.py weights/weights_evaluate_small.txt test_data/test_features_small.txt test_data/test_labels_small.txt code

runtime=$((end-start))
printf "${NC}Training time: ${RED}${runtime}${NC} seconds\n"