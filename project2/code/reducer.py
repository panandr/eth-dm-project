#!/local/anaconda/bin/python
# IMPORTANT: leave the above line as is.

import logging
import sys
import numpy as np

## REDUCER: each input line is a weight vector; finds average of all these vectors.

w_sum = np.zeros(shape=0)    # TODO initialise sum of weight vector
w_count = 0     # How many weight vectors we have summed up

def parse(line):
    """Parse line into weight vector"""
    return np.fromstring(line)


for line in sys.stdin:
    line = line.strip()

    # Parse line into weight vector
    weights = parse(line)
    print("REDUCER: Parsed line into vector of shape ", weights.shape)

    # Add weight vector to the sum
    # TODO

    w_count += 1


# Calculate average
w_result = w_sum / w_count

# Print space-separated coefficients of w_result
# TODO