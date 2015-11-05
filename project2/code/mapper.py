#!/local/anaconda/bin/python
# IMPORTANT: leave the above line as is.

import sys
import numpy as np
# from sklearn.svm import LinearSVC
# import cProfile, pstats

DIMENSION = 400          # Dimension of the original data.
# YETA = 1                # Learning Rate
# LAMDA = 1000.0           # Constraint Parameter
BATCH_SIZE = float('inf')
# BATCH_SIZE = 1000
TRANS_DIM = 800

# pr = cProfile.Profile()

# Initialise random parameters
np.random.seed(42)
omega_means = np.zeros(shape=TRANS_DIM)
omega_covariance = np.identity(TRANS_DIM)
omega = np.random.multivariate_normal(mean=omega_means, cov=omega_covariance, size=DIMENSION)
b = 2 * np.pi * np.random.rand(1, TRANS_DIM)


def permute_data(x, y):
    perm = np.random.permutation(x.shape[0])
    return x[perm, :], y[perm]
    # return perm


def transform(x_original):
    x_trans = np.sqrt(2) * np.cos(x_original.dot(omega) + b)
    return x_trans


def emit(weights):
    """Emit the weight vector from one batch of stochastic gradient descent"""
    for w in weights:
        print(str(w) + "\t"),

    print("")


def process_batch(batch, labels, weights):
    """Process a batch of examples: calculate and emit the corresponding weight vector.
    Each row in matrix 'batch' holds an example. Each element in vector 'labels' holds the corresponding label."""

    # model = LinearSVC(fit_intercept=False, C=0.1, dual=False, penalty='l1')
    # model.fit(batch, labels)
    # params = model.coef_
    # params.shape = TRANS_DIM
    for t in range(batch.shape[0]):

        YETA = 1 / (np.sqrt(t+1))
        trans_batch = (batch[t, :])

        if (labels[t]*(np.dot(weights, trans_batch))) < 1:
            weights += YETA*labels[t]*trans_batch

            a = batch.shape[0] * (1e-8)
            weights = weights * min(1, 1 / (np.sqrt(a) * np.linalg.norm(weights, 2)))
    return weights


def permutation_iter(batch, labels, weights, num_iter, min_error):
    for i in range(num_iter):
        x = np.array(batch)
        y = np.array(labels)
        [batch, labels] = permute_data(x, y)

        temp_weights = weights

        weights = process_batch(batch, labels, temp_weights)

        temp_error = np.max(np.abs(np.subtract(temp_weights, weights)))
        if (temp_error <= min_error):
            emit(weights)
            break
    emit(weights)

if __name__ == "__main__":
    # pr.enable()

    batch = np.zeros(shape=(0, TRANS_DIM))      # Initialise batch matrix
    labels = []                                 # Initialise labels list
    np.random.seed(seed=42)

    for line in sys.stdin:
        line = line.strip()
        (label, x_string) = line.split(" ", 1)
        label = int(label)
        x_original = np.fromstring(x_string, sep=' ')
        x_original.shape = (1, DIMENSION)
        x = transform(x_original)   # Use our features.
        # Add row to batch
        batch = np.concatenate((batch, x))
        labels.append(label)
        weights = np.zeros(shape=TRANS_DIM)

        num_iter = 1
        min_error = 1e-5

        # If batch has BATCH_SIZE examples then calculate weight vector on
        # that batch, process batch and start a new one
        if batch.shape[0] == BATCH_SIZE:
            num_iter = np.min([num_iter, batch.shape[0]])
            weights = process_batch(batch, labels, weights)
            permutation_iter(batch, labels, weights, num_iter, min_error)

            batch = np.zeros(shape=(0, TRANS_DIM))      # Re-initialise batch matrix
            labels = []                                 # Re-initialise labels list

    # Do last batch if any examples remain unprocessed

    if batch.shape[0] > 0:
        num_iter = np.min([num_iter, batch.shape[0]])
        weights = process_batch(batch, labels, weights)
        permutation_iter(batch, labels, weights, num_iter, min_error)
    # pr.disable()
    # ps = pstats.Stats(pr, stream=open("profile.txt", "w"))
    # ps.sort_stats("cumtime")
    # ps.print_stats()
