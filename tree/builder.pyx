# cython: linetrace=True

# (LICENSE) based on the same file as tree.pyx

from tree.tree cimport Tree

from libc.stdint cimport SIZE_MAX
from libc.stdint cimport uint32_t
from cpython.pycapsule cimport PyCapsule_IsValid, PyCapsule_GetPointer

import numpy as np
cimport numpy as np
cimport cython
np.import_array()

TREE_LEAF = -1
TREE_UNDEFINED = -2
cdef SIZE_t _TREE_LEAF = TREE_LEAF
cdef SIZE_t _TREE_UNDEFINED = TREE_UNDEFINED

cdef class Builder:
    """
    Builder builds random trees with initial_depth depth

    Args:
        initial_depth: depth of tree
    """

    def __cinit__(self):
        pass

    cpdef build(self, Tree tree, int initial_depth):
        """
        Build a random decision tree
        """
        pass

cdef class FullTreeBuilder(Builder):
    """
    FullTreeBuilder creates tree without empty spaces for nodes to the
    initial_depth depth
    """

    def __cinit__(self):
        pass

    cpdef build(self, Tree tree, int initial_depth):
        """
        Build a random decision tree
        """
        cdef SIZE_t parent = TREE_UNDEFINED
        cdef bint is_left = 0
        cdef bint is_leaf = 0
        cdef int feature
        cdef double threshold
        cdef int class_number

        cdef int rc = 0
        cdef int current_depth
        cdef int node_number
        cdef int current_node_number

        with nogil:
            for current_depth in range(initial_depth+1):
                if current_depth == initial_depth:
                    is_leaf = 1

                if rc == -1:
                    # got return code -1 - out-of-memory
                    with gil:
                        raise MemoryError()

                is_left = 0
                for node_number in range(2**current_depth):
                    if is_left == 1:
                        is_left = 0
                    else:
                        is_left = 1

                    if current_depth == 0:
                        parent = _TREE_UNDEFINED
                    else:
                        current_node_number = 2**current_depth + node_number
                        parent = <SIZE_t> ((current_node_number-2+is_left) / 2)

                    if current_depth == initial_depth:
                        class_number = tree.randint(0, tree.n_classes)
                    else:
                        feature = tree.randint(0, tree.n_features)
                        threshold = tree.thresholds[tree.randint(0, tree.n_thresholds), feature]

                    node_id = tree._add_node(parent, is_left, is_leaf, feature,
                                     threshold, current_depth, class_number)

                    if node_id == SIZE_MAX:
                        rc = -1
                        break

        tree.depth = initial_depth


cdef class SplitTreeBuilder(Builder):
    """
    FullTreeBuilder creates tree without empty spaces for nodes to the
    initial_depth depth
    """

    def __cinit__(self):
        pass

    cpdef build(self, Tree tree, int initial_depth, double split_prob = 0.7):
        """
        Build a random decision tree
        """
        cdef max_node_number = 2**initial_depth
        cdef SIZE_t parent = TREE_UNDEFINED
        cdef bint is_left = 0
        cdef bint is_leaf = 0
        cdef int feature
        cdef double threshold
        cdef int class_number

        cdef int rc = 0
        cdef int current_depth
        cdef int node_number
        cdef int right
        cdef int left

        # trzymaj tabelę nodeów, które nie są liściami i losowo lub po kolei wybieraj i zastanawiaj się czy splitować
        with nogil:
            node_number = self._node_creation(tree, initial_depth, _TREE_UNDEFINED, 0, 1, 1, split_prob)
            if node_number == -1:
                with gil:
                    raise MemoryError()

        tree.depth = node_number

    cdef int _node_creation(self, Tree tree, int initial_depth, int parent, int current_depth,
                            int is_left, int is_root, double split_prob) nogil:

        cdef int feature
        cdef double threshold
        cdef int class_number

        if current_depth == 0:
            is_leaf = 0
        elif current_depth == initial_depth:
            is_leaf = 1
        else:
            if tree.randint(0, 1000000000)/1000000000 < split_prob:
                is_leaf = 0
            else:
                is_leaf = 1

        if is_leaf == 1:
            class_number = tree.randint(0, tree.n_classes)
        else:
            feature = tree.randint(0, tree.n_features)
            threshold = tree.thresholds[tree.randint(0, tree.n_thresholds), feature]

        node_id = tree._add_node(parent, is_left, is_leaf, feature,
                         threshold, current_depth, class_number)

        if node_id == SIZE_MAX:
            rc = -1
            return -1

        if is_leaf == 0:
            right = self._node_creation(tree, initial_depth, node_id, current_depth+1, 1, 0, split_prob)
            if right == -1:
                with gil:
                    raise MemoryError()
            left = self._node_creation(tree, initial_depth, node_id, current_depth+1, 0, 0, split_prob)
            if left == -1:
                with gil:
                    raise MemoryError()
            return max(right, left)
        else:
            return current_depth

        return 0
