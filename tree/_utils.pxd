# copied from sklearn.tree._utils.pxd
# Authors: Gilles Louppe <g.louppe@gmail.com>
#          Peter Prettenhofer <peter.prettenhofer@gmail.com>
#          Arnaud Joly <arnaud.v.joly@gmail.com>
#          Jacob Schreiber <jmschreiber91@gmail.com>
#          Nelson Liu <nelson@nelsonliu.me>
#
# License: BSD 3 clause

# See _utils.pyx for details.


import numpy as np
cimport numpy as np
from tree.tree cimport Node

ctypedef np.npy_float64 DOUBLE_t         # Type of y, sample_weight
ctypedef np.npy_intp SIZE_t              # Type for indices and counters


cdef enum:
    # Max value for our rand_r replacement (near the bottom).
    # We don't use RAND_MAX because it's different across platforms and
    # particularly tiny on Windows/MSVC.
    RAND_R_MAX = 0x7FFFFFFF


ctypedef fused realloc_ptr:
    # Add pointer types here as needed.
    # commented out probably not used in our code
    # (DTYPE_t*)
    (SIZE_t*)
    # (unsigned char*)
    # (WeightedPQueueRecord*)
    (DOUBLE_t*)
    (DOUBLE_t**)
    (Node*)
    # (Cell*)
    (Node**)
    # (StackRecord*)
    # (PriorityHeapRecord*)

cdef realloc_ptr safe_realloc(realloc_ptr* p, size_t nelems) nogil except *


cdef np.ndarray sizet_ptr_to_ndarray(SIZE_t* data, SIZE_t size)
