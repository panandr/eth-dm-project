#!/local/anaconda/bin/python
# IMPORTANT: leave the above line as is.

import numpy as np
import sys


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
                # Remember that we have already detected it
                duplicates_detected.add(combination1)
                duplicates_detected.add(combination2)

last_key = None
key_count = 0
duplicates = []
duplicates_detected = set()

for line in sys.stdin:
    line = line.strip()
    key, video_id = line.split("\t")

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
