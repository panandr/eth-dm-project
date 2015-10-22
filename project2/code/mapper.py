#!/local/anaconda/bin/python
# IMPORTANT: leave the above line as is.

import sys
import numpy as np

DIMENSION = 400         # Dimension of the original data.
CLASSES = (-1, +1)      # The classes that we are trying to predict.
BATCH_SIZE = 30         # How many training examples are in each batch (Taivo set to 30 but we can change that)

def transform(x_original):
    return x_original

def emit(weights):
    """Emit the weight vector from one batch of stochastic gradient descent"""
    print(weights.tostring())

def process_batch(batch, labels):
    """Process a batch of examples: calculate and emit the corresponding weight vector.
    Each row in matrix 'batch' holds an example. Each element in vector 'labels' holds the corresponding label."""

    #print("MAPPER: Processing batch of shape %dx%d", batch.shape)

    weights = np.zeros(shape=DIMENSION)  # TODO calculate weight vector on this batch

    emit(weights)


batch = np.zeros(shape=(0, DIMENSION))      # Initialise batch matrix
labels = []                                 # Initialise labels list

for line in sys.stdin:
    line = line.strip()
    (label, x_string) = line.split(" ", 1)
    label = int(label)
    x_original = np.fromstring(x_string, sep=' ')
    x = transform(x_original)   # Use our features.
    x.shape = (1, DIMENSION)    # Force x to be a row vector

    # Add row to batch
    batch = np.concatenate((batch, x))
    labels.append(label)

    # If batch has BATCH_SIZE examples then calculate weight vector on that batch, process batch and start a new one
    if batch.shape[0] == BATCH_SIZE:
        process_batch(batch, labels)
        batch = np.zeros(shape=(0, DIMENSION))      # Re-initialise batch matrix
        labels = []                                 # Re-initialise labels list

# Do last batch if any examples remain unprocessed
if batch.shape[0] > 0:
    process_batch(batch, labels)