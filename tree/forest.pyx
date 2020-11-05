# cython: linetrace=True

from tree.tree cimport Tree

import numpy as np
cimport numpy as np

cdef class Forest:
    def __cinit__(self, int n_trees, int max_trees):
        self.trees = np.empty(max_trees, Tree)
        self.best_tree = None

        self.current_trees = 0

# ================================================================================================
# Prediction + dealloc memory
# ================================================================================================

    cpdef prepare_best_tree_to_prediction(self, int best_tree_index):
        self.best_tree = self.trees[best_tree_index]
        self.best_tree.prepare_tree_to_prediction()

    cpdef remove_other_trees(self):
        self.trees = None
        self.current_trees = 0
