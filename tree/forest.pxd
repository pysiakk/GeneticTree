from tree.tree cimport Tree
import numpy as np
cimport numpy as np

ctypedef np.npy_float64 DOUBLE_t         # Type of y
ctypedef np.npy_float32 DTYPE_t          # Type of X
ctypedef np.npy_intp SIZE_t              # Type for indices and counters

cdef class Forest:
    # Class containing all trees
    cdef public Tree[:] trees
    cdef public Tree best_tree
    cdef int best_tree_number

    cdef int n_trees
    cdef public int current_trees
    cdef int max_trees

    cdef public int n_thresholds
    cdef public DTYPE_t[:, :] thresholds

    # temporal data to use once in fit function
    cdef public object X
    cdef public np.ndarray y

    cpdef set_X_y(self, object X, np.ndarray y)
    cpdef remove_X_y(self)

    cpdef _check_input(self, object X, np.ndarray y)

    cpdef initialize_population(self, int depth)
    cdef prepare_thresholds_array(self, int n_thresholds, int n_features)

    cpdef function_to_test_nogil(self)
