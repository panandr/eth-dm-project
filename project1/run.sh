#!/usr/bin/env bash

cat data/training.txt|python2.7 code/mapper.py|sort|python2.7 code/reducer.py