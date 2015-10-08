#!/usr/bin/env bash

# Calculate duplicates according to our model
cat data/training.txt|python2.7 code/mapper.py|sort|python2.7 code/reducer.py > data/duplicates_found.txt

# Check our duplicates against provided ground truth
python2.7 code/check.py data/duplicates.txt data/duplicates_found.txt