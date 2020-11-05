# cython: linetrace=True

from tree.tree cimport Tree

import numpy as np
cimport numpy as np

ctypedef np.npy_float64 DOUBLE_t         # Type of y
ctypedef np.npy_float32 DTYPE_t          # Type of X
ctypedef np.npy_intp SIZE_t              # Type for indices and counters


cpdef DOUBLE_t[:] get_accuracies(Tree[:] trees, int current_trees):
    cdef DOUBLE_t[:] accuracies = np.empty(current_trees, float)
    cdef SIZE_t n_observations = trees[0].y.shape[0]
    cdef int i
    cdef Tree tree
    for i in range(current_trees):
        tree = trees[i]
        accuracies[i] = tree.get_proper_classified() / n_observations
    return accuracies


cpdef SIZE_t[:] get_proper_classified(Tree[:] trees, int current_trees):
    cdef SIZE_t[:] proper_classified = np.empty(current_trees, int)
    cdef int i
    cdef Tree tree
    for i in range(current_trees):
        tree = trees[i]
        proper_classified[i] = tree.get_proper_classified()
    return proper_classified


cpdef SIZE_t[:] get_trees_sizes(Tree[:] trees, int current_trees):
    cdef SIZE_t[:] trees_sizes = np.empty(current_trees, int)
    cdef int i
    for i in range(current_trees):
        trees_sizes[i] = trees[i].node_count
    return trees_sizes

