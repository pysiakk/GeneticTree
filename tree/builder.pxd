# (LICENSE) based on the same file as tree.pxd

from tree.tree cimport Tree

import numpy as np
cimport numpy as np

ctypedef np.npy_float32 DTYPE_t          # Type of X
ctypedef np.npy_float64 DOUBLE_t         # Type of y, sample_weight
ctypedef np.npy_intp SIZE_t              # Type for indices and counters
ctypedef np.npy_int32 INT32_t            # Signed 32 bit integer
ctypedef np.npy_uint32 UINT32_t          # Unsigned 32 bit integer

cdef class Builder:
    # Interface to building trees
    # cdef public int initial_depth

    # Methods
    cpdef build(self, Tree tree, int initial_depth)


