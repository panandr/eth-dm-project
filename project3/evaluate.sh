#!/usr/bin/env bash

RED='\033[0;31m'
NC='\033[0m' # No Color

echo 'Training on dataset...'
start=`date +%s`
cat data/train_10k.txt | python2.7 code/mapper.py | python2.7 code/reducer.py > centers/centers.txt
end=`date +%s`
echo 'Evaluating on 10k testset...'
printf "Score: ${RED}"
python2.7 code/evaluate.py centers/centers.txt data/test_10k.txt

runtime=$((end-start))
printf "${NC}Training time: ${RED}${runtime}${NC} seconds\n"