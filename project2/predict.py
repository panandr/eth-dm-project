import sys
import numpy as np


if len(sys.argv) < 4:
    print("Usage: python2.7 predict.py <weights filename> <features filename> <predictions filename>")
else:
    # ---- Read in weights ----
    weights_file = open(sys.argv[1], 'r')
    weights = []
    for w in weights_file.read().strip().split():
        weights.append(float(w))
    weights = np.asarray(weights)

    weights_file.close()

    # ---- Read in features ----
    features = np.genfromtxt(sys.argv[2], delimiter=' ')

    # ---- Make predictions ----
    predictions = map(lambda x: int(x) * 2 - 1, features.dot(weights) > 0)

    # ---- Save predictions to file ----
    np.savetxt(sys.argv[3], predictions, delimiter="\n", fmt="%d")