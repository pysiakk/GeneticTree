from tree.tree cimport Tree, Observation
from tree._utils cimport Stack

import numpy as np
cimport numpy as np

ctypedef np.npy_float64 DOUBLE_t         # Type of y
ctypedef np.npy_intp SIZE_t              # Type for indices and counters

cdef struct CrossoverPoint:
    SIZE_t new_parent_id
    bint is_left
    SIZE_t depth_addition

cdef class TreeCrosser:
    cpdef Tree cross_trees(self, Tree first_parent, Tree second_parent)
    cpdef Tree _cross_trees(self, Tree first_parent, Tree second_parent,
                            SIZE_t first_node_id, SIZE_t second_node_id)

    cdef _add_tree_nodes(self, Tree master, SIZE_t crossover_point,
                         Tree slave, bint is_first, CrossoverPoint* result)
    cdef create_new_observation(self, Observation observation, SIZE_t new_last_node_id)
    cdef void _register_node_in_stack(self, Tree master,
                                      SIZE_t new_parent_id, SIZE_t old_self_id,
                                      bint is_left, Stack stack) nogil

    cdef Tree _initialize_new_tree(self, Tree previous_tree)

