#!/usr/bin/env bash

RED='\033[0;31m'
NC='\033[0m' # No Color

echo 'Evaluating on LARGE dataset...'
start=`date +%s`
sh run.sh > weights/weights_evaluate.txt
end=`date +%s`
cat data/training.txt | cut -d ' ' -f1 > test_data/test_labels.txt
cat data/training.txt | cut -d ' ' -f 2- > test_data/test_features.txt
printf "Accuracy: ${RED}"
python2.7 code/evaluate.py weights/weights_evaluate.txt test_data/test_features.txt test_data/test_labels.txt code

runtime=$((end-start))
printf "${NC}Training time: ${RED}${runtime}${NC} seconds\n"