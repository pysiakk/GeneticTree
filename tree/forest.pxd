from tree.tree cimport Tree

cdef class Forest:
    # Class containing all trees
    cdef Tree[:] trees

    cpdef initial_function(self)

    cpdef function_to_test_nogil(self)
