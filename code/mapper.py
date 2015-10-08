#!/local/anaconda/bin/python
# IMPORTANT: leave the above line as is.

import numpy as np
import sys


def create_hash_functions(n):
    """Create n hash functions by generating parameters a and b"""
    a = np.random.randint(1000, size=(n, 1))
    b = np.random.randint(1000, size=(n, 1))

    return a, b


def hash(s, a, b, n_row):
    """Hash bitstring (in index form) with linear hash function"""
    a = np.tile(a, s.shape)
    b = np.tile(b, s.shape)
    n = np.tile(n_row, s.shape)
    return (np.multiply(s, a) + b) % n


def hash_band(s, a, b, n_row):
    """Hash band of signature matrix with linear hash function"""
    return (np.multiply(a, s).sum() + b) % n_row


def format_bucket(bucket1, bucket2):
    """Return a bucket string combined from the numbers of bucket 1 and bucket 2"""
    b1 = "{:05d}".format(int(bucket1))
    b2 = "{:05d}".format(int(bucket2))
    return b1 + b2


def compute_sigm(num_hash_fns, shingles):
    """Compute signature matrix column for each video"""
    # Initialise signature matrix
    SIG_M = np.zeros((num_hash_fns, 1))
        
    for i in range(num_hash_fns):
        a = hash_fns_a[i]
        b = hash_fns_b[i]
        hash_val = hash(shingles, a, b, num_row)
        SIG_M[i-1] = hash_val.min()

    return SIG_M


def emit(ls):
    """Emit list to stdout"""
    ls = map(lambda l: str(l), ls)
    print('\t'.join(ls))


def partition_sigm(num_band, SIG_M, num_hash_fns, band_id):
    """Partition the signature matrix to b bands"""
    num_rows = num_hash_fns/num_band
    parti_sigm = np.ones((num_rows, 1))
    
    for j in range(band_id * num_rows, (band_id+1) * num_rows):
        parti_sigm[j % num_rows, 0] = SIG_M[j]

    return parti_sigm


def sigm_to_string(SIG_M):
    numbers = map(lambda x: str(int(x)), SIG_M[:,0].tolist())
    return ','.join(numbers)


def test():

    # Test from lecture 3 slide 17: should print 1 2 0 0
    print(hash(np.asarray([1, 3, 4]), 1, 0, 5).min())
    print(hash(np.asarray([1, 3, 4]), 2, 1, 5).min())
    print(hash(np.asarray([2, 3, 5]), 1, 0, 5).min())
    print(hash(np.asarray([2, 3, 5]), 2, 1, 5).min())


if __name__ == "__main__":
    # VERY IMPORTANT:
    # Make sure that each machine is using the
    # same seed when generating random numbers for the hash functions.
    np.random.seed(seed=42)

    num_hash_fns = 1024     # number of hash functions
    num_band = 16           # number of bands
    num_group = 1           # number of groups for AND/OR-way

    num_row = 20001         # number of different shingles

    # Create hash functions so we use the same ones throughout the program
    hash_fns_a, hash_fns_b = create_hash_functions(num_hash_fns)

    # test()

    # Create dict to remember the parameters of each band's hash function
    band_hash_a = dict()
    band_hash_b = dict()

    for line in sys.stdin:
        # Read line and extract video ID and shingles
        line = line.strip()
        video_id = int(line[6:15])
        shingles = np.fromstring(line[16:], sep=" ")
        
        # Delete recurring elements in shingles, sort in ascending order and convert into NumPy array
        new_shingles = list(set(shingles))
        new_shingles = np.asarray(sorted(new_shingles))
        
        # Calculate signature matrix column for this video
        SIG_M = compute_sigm(num_hash_fns/num_group, new_shingles)

        # For each band
        for band_id in range(0, num_band):
            # Extract band from column
            band = partition_sigm(num_band, SIG_M, num_hash_fns/num_group, band_id)

            # Get band's hash function parameters
            if(band_id in band_hash_a):         # Band hash parameters exist
                a = band_hash_a[band_id]
                b = band_hash_b[band_id]
            else:                               # Band hash parameters don't exist and need to be created
                a, _ = create_hash_functions(band.shape[0])
                b = np.random.randint(1000)
                band_hash_a[band_id] = a
                band_hash_b[band_id] = b

            # Get hash of this band
            band_hashed = hash_band(band, a, b, num_row)

            # Get bucket ID of this band by combining the band ID and the result of hashing this band
            bucket_id = format_bucket(band_id, band_hashed)


            # Emit the bucket ID and video ID TODO: also need to emit full shingle list
            emit([bucket_id, sigm_to_string(SIG_M), video_id])
