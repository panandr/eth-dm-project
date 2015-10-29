#!/local/anaconda/bin/python
# IMPORTANT: leave the above line as is.

import sys
import numpy as np

DIMENSION = 400         # Dimension of the original data.
#YETA = 1             # Learning Rate
LAMDA = 1000.0              # Constraint Parameter
#BATCH_SIZE = 1000
NUM_CROSVALI = 10
SAMP_SUBSET = 50
BATCH_SIZE = NUM_CROSVALI * SAMP_SUBSET
TRANS_DIM = 400
#GRID_SEARCH =

def transform(x_original):
    #trans_mat = np.random.randn(DIMENSION)
    #trans_mat = np.dot(trans_mat,trans_mat)
    #np.transpose(x_original)
    temp = np.pi * np.random.randn(DIMENSION, TRANS_DIM)
    temp_omiga = np.dot(x_original, temp)
    b = np.pi * np.random.rand(1, TRANS_DIM)
    omiga = np.add(temp_omiga, b)
    x_trans = np.cos(omiga)

    return x_trans

def emit(weights):
    """Emit the weight vector from one batch of stochastic gradient descent"""
    for w in weights:
        print(str(w) + "\t"),

    print("")
#def cross_validation(batch, labels):



def process_batch(batch, labels):
    """Process a batch of examples: calculate and emit the corresponding weight vector.
    Each row in matrix 'batch' holds an example. Each element in vector 'labels' holds the corresponding label."""

    #print("MAPPER: Processing batch of shape %dx%d", batch.shape)

    weights = np.zeros(shape=TRANS_DIM)  # TODO calculate weight vector on this batch

    for ii in range(batch.shape[0]):
        YETA = 1 / (np.sqrt(ii+1))
        trans_batch = (batch[ii, :])

        if (labels[ii]*(np.dot(weights, trans_batch))) < 1:
            weights += YETA*labels[ii]*trans_batch

        weights = np.dot(np.min([1, 1 / (np.sqrt(LAMDA) * np.sqrt(np.dot(weights, weights)))]), weights)

        #return  weights
        emit(weights / batch.shape[0])

if __name__ == "__main__":
    batch = np.zeros(shape=(0, TRANS_DIM))      # Initialise batch matrix
    labels = []                                 # Initialise labels list
    np.random.seed(seed=42)

    for line in sys.stdin:
        line = line.strip()
        (label, x_string) = line.split(" ", 1)
        label = int(label)
        # x_original = np.fromstring(x_string, sep=' ')
        x_original = np.fromstring(x_string, sep=' ')
        #x.shape = (1, DIMENSION)    # Force x to be a row vector
        x_original.shape = (1, DIMENSION)
        x = transform(x_original)   # Use our features.
        # Add row to batch
        batch = np.concatenate((batch, x))
        labels.append(label)
        #print(label)

        # If batch has BATCH_SIZE examples then calculate weight vector on that batch, process batch and start a new one
        if batch.shape[0] == BATCH_SIZE:
            process_batch(batch, labels)
            batch = np.zeros(shape=(0, TRANS_DIM))      # Re-initialise batch matrix
            labels = []                                 # Re-initialise labels list

    # Do last batch if any examples remain unprocessed
    if batch.shape[0] > 0:
        process_batch(batch, labels)
