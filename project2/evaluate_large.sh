#!/usr/bin/env bash

RED='\033[0;31m'
NC='\033[0m' # No Color

echo 'Training on LARGE dataset...'
start=`date +%s`
sh run_large.sh > weights/weights_evaluate_large.txt
end=`date +%s`
# cat data/training.txt | cut -d ' ' -f1 > test_data/test_labels.txt
# cat data/training.txt | cut -d ' ' -f 2- > test_data/test_features.txt
echo 'Evaluating on MEDIUM TEST dataset...'
printf "Accuracy: ${RED}"
python2.7 code/evaluate.py weights/weights_evaluate_large.txt test_data/test_features.txt test_data/test_labels.txt code

runtime=$((end-start))
printf "${NC}Training time: ${RED}${runtime}${NC} seconds\n"