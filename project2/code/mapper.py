#!/local/anaconda/bin/python
# IMPORTANT: leave the above line as is.

import sys
import numpy as np

DIMENSION = 1 + 4*400         # Dimension of the original data.
CLASSES = (-1, +1)      # The classes that we are trying to predict.
BATCH_SIZE = 30       # How many training examples are in each batch (Taivo set to 30 but we can change that)
YETA = 1              # Learning Rate
LAMDA = 1000000000.0              # Constraint Parameter

np.random.seed(seed=42)

def transform(x_original):
    logs = np.log(1 + x_original)
    sqrts = np.sqrt(x_original)
    squares = np.square(x_original)

    # trans_mat = np.random.randn(DIMENSION)
    # trans_mat = np.dot(trans_mat,trans_mat)
    # x_original = np.log10((np.dot(10000, x_original)))
    x = np.concatenate((np.asarray([1]), x_original, logs, sqrts, squares))
    return x

def emit(weights):
    """Emit the weight vector from one batch of stochastic gradient descent"""
    for w in weights:
        print(str(w) + "\t"),

    print("")

def process_batch(batch, labels):
    """Process a batch of examples: calculate and emit the corresponding weight vector.
    Each row in matrix 'batch' holds an example. Each element in vector 'labels' holds the corresponding label."""

    weights = np.zeros(shape=DIMENSION)

    G_sum = np.ones(shape=(batch.shape[1], batch.shape[1]))

    for ii in range(batch.shape[0]):
        grad = labels[ii] * batch[ii, :]
        grad.shape = (grad.shape[0], 1)
        G_sum += grad.dot(grad.T)
        G_t_diag = np.sqrt(np.diag(G_sum))
        G_t_diag_inv = 1 / G_t_diag
        G_t_inv = np.multiply(np.identity(grad.shape[0]), G_t_diag_inv)

        grad.shape = grad.shape[0]
        if (np.dot(weights, grad)) < 1:
            weights += YETA * G_t_inv.dot(grad)

    weights = np.dot(np.min([1, 1 / (np.sqrt(LAMDA) * np.sqrt(np.dot(weights, weights)))]), weights)
    emit(weights)

if __name__ == "__main__":
    batch = np.zeros(shape=(0, DIMENSION))      # Initialise batch matrix
    labels = []                                 # Initialise labels list
    np.random.seed(seed=42)

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
