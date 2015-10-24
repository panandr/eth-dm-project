#!/local/anaconda/bin/python
# IMPORTANT: leave the above line as is.

import sys
import numpy as np

DIMENSION = 400         # Dimension of the original data.
CLASSES = (-1, +1)      # The classes that we are trying to predict.
BATCH_SIZE = 200       # How many training examples are in each batch (Taivo set to 30 but we can change that)
# YETA = 0.2              # Learning Rate
LAMDA = 1               # Constraint Parameter


def transform(x_original):
    return x_original

def emit(weights):
    """Emit the weight vector from one batch of stochastic gradient descent"""
    for w in weights:
        print(str(w) + "\t"),

    print("")

def process_batch(batch, labels):
    """Process a batch of examples: calculate and emit the corresponding weight vector.
    Each row in matrix 'batch' holds an example. Each element in vector 'labels' holds the corresponding label."""

    #print("MAPPER: Processing batch of shape %dx%d", batch.shape)

    weights = np.zeros(shape=DIMENSION)  # TODO calculate weight vector on this batch

    for ii in range(batch.shape[0]):
        YETA = 1 / (np.sqrt(ii+1))
        if (labels[ii]*(np.dot(weights, batch[ii,:]))) < 1:
            weights += YETA*labels[ii]*batch[ii,:]
            weights = np.min([1, 1 / np.sqrt(np.dot(weights, weights))]) * weights
    emit(weights / batch.shape[0])          # Divide by number of rows so we can simply sum in mapper
    #faster convergence
#            temp_s=1
#            for dim in range(DIMENSION):
#                for jj in range(ii):
#                    temp_s += np.dot(-labels[jj]*batch[ii,:],-labels[jj]*batch[ii,:])
#                weights[dim] += -(YETA*labels[ii]*batch[ii,:]) / (sqrt(temp_s))
        
    #emit(weights / batch.shape[0])

if __name__ == "__main__":
    batch = np.zeros(shape=(0, DIMENSION))      # Initialise batch matrix
    labels = []                                 # Initialise labels list

    for line in sys.stdin:
        line = line.strip()
        (label, x_string) = line.split(" ", 1)
        label = int(label)
        # x_original = np.fromstring(x_string, sep=' ')
        x = np.fromstring(x_string, sep=' ')
        # x = transform(x_original)   # Use our features.
        x.shape = (1, DIMENSION)    # Force x to be a row vector

        # Add row to batch
        batch = np.concatenate((batch, x))
        labels.append(label)
        #print(label)

        # If batch has BATCH_SIZE examples then calculate weight vector on that batch, process batch and start a new one
        if batch.shape[0] == BATCH_SIZE:
            process_batch(batch, labels)
            batch = np.zeros(shape=(0, DIMENSION))      # Re-initialise batch matrix
            labels = []                                 # Re-initialise labels list

    # Do last batch if any examples remain unprocessed
    if batch.shape[0] > 0:
        process_batch(batch, labels)
