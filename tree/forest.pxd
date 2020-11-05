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
    cdef int best_tree_index

    cdef int n_trees
    cdef public int current_trees
    cdef int max_trees

    cdef public int n_thresholds
    cdef public DTYPE_t[:, :] thresholds

    # temporal data to use once in fit function
    cdef public object X
    cdef public np.ndarray y
    cdef public int n_features
    cdef public int n_classes

    # Set up
    cpdef set_X_y(self, object X, np.ndarray y)
    cpdef prepare_thresholds_array(self)

    # Initializer
    cpdef add_new_tree_and_initialize_observations(self, Tree tree)

    # Prediction + dealloc memory
    cpdef prepare_best_tree_to_prediction(self, int best_tree_index)
    cpdef remove_unnecessary_variables(self)
    cpdef __remove_X_y__(self)
    cpdef remove_other_trees(self)

    cpdef np.ndarray predict(self, object X)
