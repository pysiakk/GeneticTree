# cython: linetrace=True

# (LICENSE) based on the same file as tree.pyx

from tree.tree cimport Tree

from libc.stdint cimport SIZE_MAX
from libc.stdint cimport uint32_t
from cpython.pycapsule cimport PyCapsule_IsValid, PyCapsule_GetPointer

import numpy as np
cimport numpy as np
import cython
from numpy.random cimport bitgen_t
from numpy.random import PCG64
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

    def __cinit__(self, int initial_depth):
        self.initial_depth = initial_depth

    cpdef build(self, Tree tree):
        """
        Build a random decision tree
        """
        pass

cdef class FullTreeBuilder(Builder):
    """
    FullTreeBuilder creates tree without empty spaces for nodes to the
    initial_depth depth
    """

    def __cinit__(self, int initial_depth):
        self.initial_depth = initial_depth

    cpdef build(self, Tree tree):
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

        # declaration to use nogil random generator (self.bounded_uint)
        cdef bitgen_t *rng
        cdef const char *capsule_name = "BitGenerator"
        x = PCG64(np.random.randint(0, 10**8))
        capsule = x.capsule
        if not PyCapsule_IsValid(capsule, capsule_name):
            raise ValueError("Invalid pointer to anon_func_state")
        rng = <bitgen_t *>PyCapsule_GetPointer(capsule, capsule_name)

        with nogil:
            for current_depth in range(self.initial_depth+1):
                if current_depth == self.initial_depth:
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

                    if current_depth == self.initial_depth:
                        class_number = self.bounded_uint(0, tree.n_classes-1, rng)
                    else:
                        feature = self.bounded_uint(0, tree.n_features-1, rng)
                        threshold = tree.thresholds[self.bounded_uint(0, tree.n_thresholds-1, rng), feature]

                    node_id = tree._add_node(parent, is_left, is_leaf, feature,
                                     threshold, current_depth, class_number)

                    if node_id == SIZE_MAX:
                        rc = -1
                        break

        tree.depth = self.initial_depth

    # function and usage from https://numpy.org/devdocs/reference/random/examples/cython/extending.pyx.html
    cdef SIZE_t bounded_uint(self, SIZE_t lb, SIZE_t ub, bitgen_t *rng) nogil:
        cdef SIZE_t mask, delta, val
        mask = delta = ub - lb
        mask |= mask >> 1
        mask |= mask >> 2
        mask |= mask >> 4
        mask |= mask >> 8
        mask |= mask >> 16

        val = rng.next_uint32(rng.state) & mask
        while val > delta:
            val = rng.next_uint32(rng.state) & mask

        return lb + val
