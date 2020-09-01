import numpy as np
cimport numpy as np
from tree.tree cimport Tree

cdef class Forest:
    property trees:
        def __get__(self):
            return self.trees

    def __cinit__(self, int max_trees):
        self.trees = np.empty(max_trees, Tree)
        self.initial_function()

    # initialize some trees to not nulls
    cpdef initial_function(self):
        # print("Run initial_function of TreeContainer")
        # create some random objects
        for i in range(4):
            self.trees[i] = Tree(5, np.zeros(2, dtype=np.int), 1)
        # print one of them
        # print(self.trees[0])

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
