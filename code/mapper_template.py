#!/local/anaconda/bin/python
# IMPORTANT: leave the above line as is.

import numpy as np
import sys

def hash(s, a, b):
    return np.dot(s, a) + b % 9907

# l is a list, will be emitted with tab separation
def emit(ls):
    ls = map(lambda l: str(l), ls)
    print('\t'.join(ls))

if __name__ == "__main__":
    # VERY IMPORTANT:
    # Make sure that each machine is using the
    # same seed when generating random numbers for the hash functions.
    np.random.seed(seed=42)

    for line in sys.stdin:
        print line
        line = line.strip()
        video_id = int(line[6:15])
        shingles = np.fromstring(line[16:], sep=" ")





