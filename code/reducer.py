#!/local/anaconda/bin/python
# IMPORTANT: leave the above line as is.

import numpy as np
import sys

def sigm_string_to_sigm(sigm_string):
    """Turn comma-separated signature matrix column into Numpy array."""
    return map(lambda x: int(x), sigm_string.split(','))

def sigm_similarity(sigm_string1, sigm_string2):
    """Calculate proportion of bits that are equal in sigm_string1 and sigm_string2"""
    ind1 = sigm_string_to_sigm(sigm_string1)
    ind2 = sigm_string_to_sigm(sigm_string2)

    bin1 = np.zeros(20001, dtype=bool)
    bin2 = np.zeros(20001, dtype=bool)

    bin1[ind1] = 1
    bin2[ind2] = 1

    def similarity(bin1, bin2):
        """Calculate proportion of bits that are equal in bin1 and bin2; assumes equal length of vectors."""
        zipped = zip(bin1, bin2)
        return sum(map(lambda x: x[0] == x[1], zipped)) / float(len(bin1))

    return similarity(bin1, bin2)


def print_duplicates(videos):
    unique = np.unique(videos)
    for i in xrange(len(unique)):
        for j in xrange(i + 1, len(unique)):
            video1 = unique[i]
            video2 = unique[j]
            combination1 = str(video1) + str(video2)
            combination2 = str(video2) + str(video1)
            # Check if video has been already detected as duplicate
            if combination1 not in duplicates_detected and combination2 not in duplicates_detected:
                # Print duplicate
                print "%d\t%d" % (min(unique[i], unique[j]),
                                  max(unique[i], unique[j]))
                print(sigm_similarity(video_sigm[video1], video_sigm[video2]))

                # Remember that we have already detected it
                duplicates_detected.add(combination1)
                duplicates_detected.add(combination2)

last_key = None
key_count = 0
duplicates = []
duplicates_detected = set()
video_sigm = dict()

for line in sys.stdin:
    line = line.strip()
    key, sigm_string, video_id = line.split("\t")
    video_sigm[int(video_id)] = sigm_string

    if last_key is None:
        last_key = key

    if key == last_key:
        duplicates.append(int(video_id))
    else:
        # Key changed (previous line was k=x, this line is k=y)
        #print("Reducer at key: {}, count of duplicates: {}".format(key, len(duplicates)))
        print_duplicates(duplicates)
        duplicates = [int(video_id)]
        last_key = key

if len(duplicates) > 0:
    print_duplicates(duplicates)
