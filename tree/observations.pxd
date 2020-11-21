import numpy as np
cimport numpy as np
np.import_array()

ctypedef np.npy_float64 DOUBLE_t        # Type of thresholds
ctypedef np.npy_float32 DTYPE_t         # Type of X
ctypedef np.npy_intp SIZE_t             # Type for indices and counters

from tree._utils cimport IntArray, Leaves

cdef class Observations:
    cdef Leaves leaves

    cdef public proper_classified
    cdef public SIZE_t n_observations   # Number of observations in X and y

    cdef public object X                    # Array with observations features (TODO: possibility of sparse array)
    cdef DTYPE_t[:, :] X_ndarray
    cdef public SIZE_t[:] y                 # Array with classes of observations

    cpdef initialize_observations(self)
    cpdef remove_observations(self, SIZE_t leaves_id)

    cpdef reassign_observations(self, SIZE_t below_node_id)
    cdef _reassign_observations_for_leaf(self, SIZE_t leaves_id, SIZE_t below_node_id)
    cdef _assign_observation(self, SIZE_t y_id, SIZE_t below_node_id)

    cdef SIZE_t _append_leaves(self, SIZE_t y_id)        # return leaves_id
    cdef _append_observations(self, SIZE_t leaves_id, SIZE_t y_id)

    cpdef test_create_leaves_array_simple(self)
    cpdef test_create_leaves_array_complex(self)
    cpdef test_create_leaves_array_many(self)
