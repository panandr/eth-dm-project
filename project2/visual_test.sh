#!/usr/bin/env bash

echo 'Training SVM...'
sh run_large.sh > weights/weights1.txt

echo 'Predicting...'
python2.7 predict.py weights/weights1.txt visual_test/visual_test_set.csv visual_test/prediction.csv
cd visual_test
python2.7 visual_test.py

echo 'Opening resulting HTML file...'
open results.html
cd ..