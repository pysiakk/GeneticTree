import numpy as np
cimport numpy as np
from tree.tree cimport Tree
from tree.builder cimport Builder
from tree.builder import FullTreeBuilder

cdef class Forest:
    property trees:
        def __get__(self):
            return self.trees

    property best_tree:
        def __get__(self):
            return self.trees[self.best_tree_number]

    def __cinit__(self, int n_trees, int max_trees):
        self.trees = np.empty(max_trees, Tree)
        self.best_tree_number = 0

        self.n_trees = n_trees
        self.max_trees = max_trees
        self.current_trees = 0

    cpdef set_X_y(self, object X, np.ndarray y):
        self.X = X
        self.y = y

    cpdef remove_X_y(self):
        self.X = None
        self.y = None

    # basic initialization function
    cpdef initialize_population(self, int max_depth):
        # TODO initialize this parameters with X and y
        cdef int n_features = self.X.shape[1]
        cdef np.ndarray[SIZE_t, ndim=1] n_classes = np.zeros(1, dtype=np.int)
        cdef int n_outputs = 1

        cdef Builder builder = FullTreeBuilder(max_depth)

        for i in range(self.n_trees):
            self.trees[i] = Tree(n_features, n_classes, n_outputs)
            builder.build(self.trees[i], self.X, self.y)
            self.current_trees += 1

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
