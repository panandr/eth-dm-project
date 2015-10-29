#!/local/anaconda/bin/python
# IMPORTANT: leave the above line as is.

import logging
import sys
import numpy as np

## Reducer: each input line is a weight vector; finds average of all these vectors.

DIMENSION = 400

w_sum = np.zeros(shape=DIMENSION)
w_count = 0     # How many weight vectors we have summed up

def parse(line):
    """Parse line into weight vector"""
    line = line.strip().split("\t")
    weights = []
    for w in line:
        weights.append(float(w))

    return np.asarray(weights)

def emit(weights):
    """Emit the weight vector from one batch of stochastic gradient descent"""
    for w in weights:
        print(str(w) + " "),

    #print("")

for line in sys.stdin:
    line = line.strip()

    # Parse line into weight vector
    weights = parse(line)

    # Add weight vector to the sum
    w_sum += weights
    w_count += 1


# Print space-separated coefficients of w_result
weights = w_sum / w_count
emit(weights)
