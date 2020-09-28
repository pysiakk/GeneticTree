from tree.tree cimport Tree
from tree._utils cimport Stack

import numpy as np
cimport numpy as np

ctypedef np.npy_float64 DOUBLE_t         # Type of y
ctypedef np.npy_intp SIZE_t              # Type for indices and counters

"""
Structure to define handy values in order to copy nodes from second parent to child
"""
cdef struct CrossoverPoint:
    SIZE_t new_parent_id        # id of node that should be a parent of first copied node from second tree
    bint is_left                # if copied node should be registered as left
    SIZE_t depth_addition       # what is the depth of copied node in child tree

cdef class TreeCrosser:
    cpdef Tree cross_trees(self, Tree first_parent, Tree second_parent)
    cpdef Tree _cross_trees(self, Tree first_parent, Tree second_parent,
                            SIZE_t first_node_id, SIZE_t second_node_id)

    cdef _copy_nodes(self, Tree donor, SIZE_t crossover_point,
                     Tree recipient, bint is_first, CrossoverPoint* result)
    cdef void _add_node_to_stack(self, Tree donor,
                                 SIZE_t new_parent_id, SIZE_t old_self_id,
                                 bint is_left, Stack stack) nogil

    cdef Tree _initialize_new_tree(self, Tree previous_tree)

