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

from scipy.sparse import issparse

from numpy import float32 as DTYPE
from numpy import float64 as DOUBLE

TREE_LEAF = -1
TREE_UNDEFINED = -2
cdef SIZE_t _TREE_LEAF = TREE_LEAF
cdef SIZE_t _TREE_UNDEFINED = TREE_UNDEFINED

cdef class Builder:

    cpdef build(self, Tree tree, object X, np.ndarray y):
        """Build a decision tree from the training set (X, y)."""
        pass

    # TODO move check input to forest to do it only once
    cdef inline _check_input(self, object X, np.ndarray y):
        """Check input dtype, layout and format"""
        if issparse(X):
            X = X.tocsc()
            X.sort_indices()

            if X.data.dtype != DTYPE:
                X.data = np.ascontiguousarray(X.data, dtype=DTYPE)

            if X.indices.dtype != np.int32 or X.indptr.dtype != np.int32:
                raise ValueError("No support for np.int64 index based "
                                 "sparse matrices")

        elif X.dtype != DTYPE:
            # since we have to copy we will make it fortran for efficiency
            X = np.asfortranarray(X, dtype=DTYPE)

        if y.dtype != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        return X, y

cdef class FullTreeBuilder(Builder):
    """Build a full random tree."""

    def __cinit__(self, int depth):
        self.depth = depth

    cpdef build(self, Tree tree, object X, np.ndarray y):
        """Build a decision tree from the training set (X, y)."""

        # check input
        X, y = self._check_input(X, y)

        # Initial capacity
        cdef int init_capacity

        if tree.max_depth <= 10:
            init_capacity = (2 ** (tree.max_depth + 1)) - 1
        else:
            init_capacity = 2047

        tree._resize(init_capacity)

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
        x = PCG64()
        capsule = x.capsule
        if not PyCapsule_IsValid(capsule, capsule_name):
            raise ValueError("Invalid pointer to anon_func_state")
        rng = <bitgen_t *>PyCapsule_GetPointer(capsule, capsule_name)

        with nogil:
            for current_depth in range(self.depth+1):
                if current_depth == self.depth:
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

                    #TODO get random feature, threshold or class
                    if current_depth == self.depth:
                        class_number = 0
                        # class_number = self.bounded_uint(0, tree.n_classes, rng)
                    else:
                        feature = self.bounded_uint(0, tree.n_features, rng)
                        threshold = 0.0

                    node_id = tree._add_node(parent, is_left, is_leaf, feature,
                                     threshold, current_depth, class_number)

                    if node_id == SIZE_MAX:
                        rc = -1
                        break

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
