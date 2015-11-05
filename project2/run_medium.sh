#!/usr/bin/env bash

cat data/training_medium.txt|python2.7 code/mapper.py|python2.7 code/reducer.py