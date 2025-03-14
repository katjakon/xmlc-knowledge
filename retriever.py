import numpy as np
import faiss  

d = 64                           # dimension of embeddings.  
M = 32                           # dimension, higher more accurate but more memory.
nb = 100000                      # database size, should be n_labels.
nq = 10000                       # nb of queries, should be n_docs
np.random.seed(1234)             # make reproducible

# Replace this by some pre-trained representations of labels.
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000.
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.

# make faiss available
index = faiss.IndexHNSWFlat(d, M)   # build the index
print(index.is_trained)
index.add(xb)                  # add vectors to the index
print(index.ntotal)