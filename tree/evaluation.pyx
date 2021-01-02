# cython: linetrace=True

from tree.tree cimport Tree

import numpy as np
cimport numpy as np

ctypedef np.npy_float64 DOUBLE_t         # Type of y
ctypedef np.npy_float32 DTYPE_t          # Type of X
ctypedef np.npy_intp SIZE_t              # Type for indices and counters


cpdef DOUBLE_t[:] get_accuracies(list trees):
    cdef DOUBLE_t[:] accuracies = np.empty(len(trees), float)
    cdef DTYPE_t sample_weight_sum = np.sum(trees[0].sample_weight)
    cdef int i
    cdef Tree tree
    for i in range(len(trees)):
        tree = trees[i]
        accuracies[i] = tree.proper_classified / sample_weight_sum
    return accuracies


cpdef SIZE_t[:] get_proper_classified(list trees):
    cdef SIZE_t[:] proper_classified = np.empty(len(trees), int)
    cdef int i
    cdef Tree tree
    for i in range(len(trees)):
        tree = trees[i]
        proper_classified[i] = tree.proper_classified
    return proper_classified


cpdef SIZE_t[:] get_trees_n_leaves(list trees):
    cdef SIZE_t[:] trees_sizes = np.empty(len(trees), int)
    cdef int i
    for i in range(len(trees)):
        trees_sizes[i] = (trees[i].node_count + 1) / 2
    return trees_sizes


cpdef SIZE_t[:] get_trees_sizes(list trees):
    cdef SIZE_t[:] trees_sizes = np.empty(len(trees), int)
    cdef int i
    for i in range(len(trees)):
        trees_sizes[i] = trees[i].node_count
    return trees_sizes



cpdef SIZE_t[:] get_trees_depths(list trees):
    cdef SIZE_t[:] trees_depths = np.empty(len(trees), int)
    cdef int i
    for i in range(len(trees)):
        trees_depths[i] = trees[i].depth
    return trees_depths
