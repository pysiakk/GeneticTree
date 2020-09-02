import numpy as np
cimport numpy as np
from tree.tree cimport Tree
from tree.builder cimport Builder

cdef class Forest:
    property trees:
        def __get__(self):
            return self.trees

    property best_tree:
        def __get__(self):
            return self.trees[self.best_tree_number]

    def __cinit__(self, int population_size, int max_trees):
        self.trees = np.empty(max_trees, Tree)
        self.best_tree_number = 0

        self.population_size = population_size
        self.max_trees = max_trees
        self.current_trees = 0

    # basic initialization function
    cpdef initialize_population(self, object X, np.ndarray y, int depth):
        self.X = X
        self.y = y

        # TODO initialize this parameters with X and y
        cdef int n_features = 5
        cdef np.ndarray[SIZE_t, ndim=1] n_classes = np.zeros(1, dtype=np.int)
        cdef int n_outputs = 1

        cdef Builder builder = Builder(depth)

        for i in range(self.population_size):
            self.trees[i] = Tree(n_features, n_classes, n_outputs)
            builder.build(self.trees[i], X, y)
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