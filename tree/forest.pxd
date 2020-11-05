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

    cdef public int current_trees

    # Prediction + dealloc memory
    cpdef prepare_best_tree_to_prediction(self, int best_tree_index)
    cpdef remove_other_trees(self)
