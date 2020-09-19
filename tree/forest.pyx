from tree.tree cimport Tree
from tree.builder cimport Builder
from tree.builder import FullTreeBuilder

import numpy as np
cimport numpy as np
from scipy.sparse import issparse

from numpy import float32 as DTYPE
from numpy import float64 as DOUBLE

cdef class Forest:
    def __cinit__(self, int n_trees, int max_trees, int n_thresholds):
        self.trees = np.empty(max_trees, Tree)
        self.best_tree_number = 0

        self.n_trees = n_trees
        self.max_trees = max_trees
        self.current_trees = 0

        self.n_thresholds = n_thresholds

    cpdef set_X_y(self, object X, np.ndarray y):
        X, y = self._check_input(X, y)
        self.X = X
        self.y = y

    cpdef remove_X_y(self):
        self.X = None
        self.y = None

    cpdef _check_input(self, object X, np.ndarray y):
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


    # basic initialization function
    cpdef initialize_population(self, int initial_depth):
        cdef int n_features = self.X.shape[1]
        cdef int n_classes = np.unique(self.y).shape[0]

        cdef Builder builder = FullTreeBuilder(initial_depth)

        self.prepare_thresholds_array(self.n_thresholds, n_features)

        for i in range(self.n_trees):
            self.trees[i] = Tree(n_features, n_classes, self.thresholds, initial_depth)
            builder.build(self.trees[i], self.X, self.y)
            self.current_trees += 1

    cdef prepare_thresholds_array(self, int n_thresholds, int n_features):
        cdef DTYPE_t[:, :] thresholds = np.zeros([n_thresholds, n_features], dtype=DTYPE)
        cdef int i
        cdef int j
        cdef int index
        cdef DTYPE_t[:, :] X_ndarray = self.X
        cdef DTYPE_t[:] X_column

        for i in range(n_features):
            X_column = X_ndarray[:, i]
            X_column = np.sort(X_column)
            for j in range(n_thresholds):
                index = int(X_column.shape[0] / (n_thresholds+1) * (j+1))
                thresholds[j, i] = X_column[index]
        self.thresholds = thresholds

    # main function to test how to process many trees in one time using few cores
    cpdef function_to_test_nogil(self):
        cdef int i
        cdef Tree[:] trees = self.trees
        print("Start testing nogil")
        with nogil:
            for i in range(4):
                i = i+1
                # below code is not working with nogil :(
                # the compilation can't end successfully
                # trees[i].time_test_function()
        print("Nogil tested successfully")
