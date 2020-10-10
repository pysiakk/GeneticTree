# cython: linetrace=True

from tree.tree cimport Tree

import numpy as np
cimport numpy as np

from numpy import float32 as DTYPE

cdef class Forest:
    def __cinit__(self, int n_trees, int max_trees, int n_thresholds):
        self.trees = np.empty(max_trees, Tree)
        self.best_tree_index = 0
        self.best_tree = None

        self.n_trees = n_trees
        self.max_trees = max_trees
        self.current_trees = 0

        self.n_thresholds = n_thresholds
        self.thresholds = None

        self.X = None
        self.y = None

# ================================================================================================
# Set up
# ================================================================================================

    cpdef set_X_y(self, object X, np.ndarray y):
        self.X = X
        self.y = y
        self.n_features = self.X.shape[1]
        self.n_classes = np.unique(self.y).shape[0]

    cpdef prepare_thresholds_array(self):
        cdef DTYPE_t[:, :] thresholds = np.zeros([self.n_thresholds, self.n_features], dtype=DTYPE)
        cdef int i
        cdef int j
        cdef int index
        cdef DTYPE_t[:, :] X_ndarray = self.X
        cdef DTYPE_t[:] X_column

        for i in range(self.n_features):
            X_column = X_ndarray[:, i]
            X_column = np.sort(X_column)
            for j in range(self.n_thresholds):
                index = int(X_column.shape[0] / (self.n_thresholds+1) * (j+1))
                thresholds[j, i] = X_column[index]
        self.thresholds = thresholds

# ================================================================================================
# Initializer
# ================================================================================================

    cpdef create_new_tree(self, int initial_depth):
        return Tree(self.n_features, self.n_classes, self.thresholds, initial_depth)

    cpdef add_new_tree_and_initialize_observations(self, Tree tree):
        tree.initialize_observations(self.X, self.y)
        self.trees[self.current_trees] = tree
        self.current_trees += 1

# ===========================================================================================================
# Evaluation functions
# ===========================================================================================================

    cpdef DOUBLE_t[:] get_accuracies(self):
        cdef DOUBLE_t[:] accuracies = np.empty(self.current_trees, float)
        cdef SIZE_t n_observations = self.y.shape[0]
        cdef int i
        cdef Tree tree
        for i in range(self.current_trees):
            tree = self.trees[i]
            accuracies[i] = tree.get_proper_classified(self.X) / n_observations
        return accuracies

    cpdef SIZE_t[:] get_proper_classified(self):
        cdef SIZE_t[:] proper_classified = np.empty(self.current_trees, int)
        cdef int i
        cdef Tree tree
        for i in range(self.current_trees):
            tree = self.trees[i]
            proper_classified[i] = tree.get_proper_classified(self.X)
        return proper_classified

    cpdef SIZE_t[:] get_trees_sizes(self):
        cdef SIZE_t[:] trees_sizes = np.empty(self.current_trees, int)
        cdef int i
        for i in range(self.current_trees):
            trees_sizes[i] = self.trees[i].node_count
        return trees_sizes

# ================================================================================================
# Prediction + dealloc memory
# ================================================================================================

    cpdef prepare_best_tree_to_prediction(self, int best_tree_index):
        self.best_tree_index = best_tree_index
        self.best_tree = self.trees[best_tree_index]
        self.best_tree.prepare_tree_to_prediction()

    cpdef remove_unnecessary_variables(self):
        self.__remove_X_y__()
        self.thresholds = None

    cpdef __remove_X_y__(self):
        self.X = None
        self.y = None

    cpdef remove_other_trees(self):
        self.trees = None
        self.current_trees = 0

    cpdef np.ndarray predict(self, object X):
        return self.best_tree.predict(X)
