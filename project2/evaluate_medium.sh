#!/usr/bin/env bash

RED='\033[0;31m'
NC='\033[0m' # No Color

echo 'Training on MEDIUM dataset...'
start=`date +%s`
sh run_medium.sh > weights/weights_evaluate_medium.txt
end=`date +%s`
echo 'Evaluating on MEDIUM TEST dataset...'
printf "Accuracy: ${RED}"
python2.7 code/evaluate.py weights/weights_evaluate_medium.txt test_data/test_features.txt test_data/test_labels.txt code

runtime=$((end-start))
printf "${NC}Training time: ${RED}${runtime}${NC} seconds\n"