#!/local/anaconda/bin/python
# IMPORTANT: leave the above line as is.

import numpy as np
import sys

##add the path of dataset
##sys.path.append('e:\\Py_Prac\\Project_1')


def hash(s, a, b):
    n_row=20001
    return (np.dot(s, a) + b) % n_row

##function to compute signature matrix for each video
def compute_sigm(num_hash_fns,shingles):
    #initial of signature matrix
    SIG_M=[0 for i in range(num_hash_fns)]
        
    for n in range(num_hash_fns):
        a=np.random.randint(1000)
        b=np.random.randint(1000)
        hash_val=hash(new_shingles,a,b)
        SIG_M[n-1]=min(hash_val)
    return SIG_M

# l is a list, will be emitted with tab separation
def emit(ls):
    ls = map(lambda l: str(l), ls)
    print('\t'.join(ls))

#partition the signature matrix to b bands
def partition_sigm(num_band,SIG_M,num_hash_fns,band_id):
    num_col=num_hash_fns/num_band
    parti_sigm=np.ones(num_col)
    
    for j in range(band_id*(num_col),(band_id+1)*(num_col)):    
        parti_sigm[j % num_col]=SIG_M[j]
    return parti_sigm

if __name__ == "__main__":
    # VERY IMPORTANT:
    # Make sure that each machine is using the
    # same seed when generating random numbers for the hash functions.
    np.random.seed(seed=42)
    #load the training dataset
    input_file=open('training.txt','r')

    num_hash_fns=1024;   #define number of hash functions
    num_band=16;       #define number of bands
    num_group=1;      #define number of groups for AND/OR-way
    
    for line in sys.stdin:
    #for line in input_file:     #just for local test
        #print line
        line = line.strip()

        #read the id of video
        video_id = int(line[6:15])

        #read the shingles from each line
        shingles = np.fromstring(line[16:], sep=" ")
        
        # delete the recur element in shingles
        #and sort in ascending order
        new_shingles=list(set(shingles))
        new_shingles=sorted(new_shingles)
        
        #SIG_M=np.array((1,num_hash_fns/num_band,num_group))        
        SIG_M=compute_sigm(num_hash_fns/num_group,new_shingles)
        
        
        for band_id in range(0, num_band):
            parti_SIG_M=partition_sigm(num_band,SIG_M,num_hash_fns/num_group,band_id)
            emit([band_id,parti_SIG_M,video_id])
        
