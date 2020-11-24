# cython: linetrace=True

# copied from sklearn.tree._utils.pyx
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# Authors: Gilles Louppe <g.louppe@gmail.com>
#          Peter Prettenhofer <peter.prettenhofer@gmail.com>
#          Arnaud Joly <arnaud.v.joly@gmail.com>
#          Jacob Schreiber <jmschreiber91@gmail.com>
#          Nelson Liu <nelson@nelsonliu.me>
#
# In context of authors there are copied methods from tree.pyx
#
# License: BSD 3 clause

from libc.stdlib cimport free
from libc.stdlib cimport malloc
from libc.stdlib cimport realloc
from libc.stdint cimport SIZE_MAX

import numpy as np
cimport numpy as np
np.import_array()


# =============================================================================
# Helper functions
# =============================================================================

cdef realloc_ptr safe_realloc(realloc_ptr* p, size_t nelems) nogil except *:
    # sizeof(realloc_ptr[0]) would be more like idiomatic C, but causes Cython
    # 0.20.1 to crash.
    cdef size_t nbytes = nelems * sizeof(p[0][0])
    if nbytes / sizeof(p[0][0]) != nelems:
        # Overflow in the multiplication
        with gil:
            raise MemoryError("could not allocate (%d * %d) bytes"
                              % (nelems, sizeof(p[0][0])))
    cdef realloc_ptr tmp = <realloc_ptr>realloc(p[0], nbytes)
    if tmp == NULL:
        with gil:
            raise MemoryError("could not allocate %d bytes" % nbytes)

    p[0] = tmp
    return tmp  # for convenience

cdef int resize(DynamicArray* array, SIZE_t capacity) nogil except -1:
        """Resize all inner arrays to `capacity`, if `capacity` == -1, then
           double the size of the inner arrays.
        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        if resize_c(array, capacity) != 0:
            # Acquire gil only if we need to raise
            with gil:
                raise MemoryError()


cdef int resize_c(DynamicArray* array, SIZE_t capacity=SIZE_MAX) nogil except -1:
        """Guts of _resize
        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        if capacity == array[0].capacity and array[0].elements != NULL:
            return 0

        if capacity == SIZE_MAX:
            if array[0].capacity == 0:
                capacity = 3  # default initial value
            else:
                capacity = 2 * array[0].capacity

        safe_realloc(&array[0].elements, capacity)

        # if capacity smaller than node_count, adjust the counter
        if capacity < array[0].count:
            array[0].count = capacity

        array[0].capacity = capacity
        return 0

def _realloc_test():
    # Helper for tests. Tries to allocate <size_t>(-1) / 2 * sizeof(size_t)
    # bytes, which will always overflow.
    cdef SIZE_t* p = NULL
    safe_realloc(&p, <size_t>(-1) / 2)
    if p != NULL:
        free(p)
        assert False


cdef inline np.ndarray sizet_ptr_to_ndarray(SIZE_t* data, SIZE_t size):
    """Return copied data as 1D numpy array of intp's."""
    cdef np.npy_intp shape[1]
    shape[0] = <np.npy_intp> size
    return np.PyArray_SimpleNewFromData(1, shape, np.NPY_INTP, data).copy()


cdef IntArray copy_int_array(IntArray* old_array):
    cdef IntArray new_array
    new_array.capacity = 0
    new_array.count = 0
    new_array.elements = NULL
    resize_c(&new_array, old_array.count)
    cdef SIZE_t i
    for i in range(old_array.count):
        new_array.elements[i] = old_array.elements[i]
    new_array.count = old_array.count
    return new_array

cdef Leaves copy_leaves(Leaves* old_leaves):
    cdef Leaves new_leaves
    new_leaves.capacity = 0
    new_leaves.count = 0
    new_leaves.elements = NULL
    resize_c(&new_leaves, old_leaves.count)
    cdef SIZE_t i
    for i in range(old_leaves.count):
        new_leaves.elements[i] = copy_int_array(&old_leaves.elements[i])
    new_leaves.count = old_leaves.count
    return new_leaves


cdef IntArray create_int_array(SIZE_t factor):
    cdef IntArray int_array
    int_array.capacity = 0
    int_array.count = 0
    int_array.elements = NULL
    resize_c(&int_array, 10)
    cdef SIZE_t i
    for i in range(10):
        int_array.elements[i] = i * factor + 1
    int_array.count = 10
    return int_array

cdef Leaves create_leaves():
    cdef Leaves leaves
    leaves.capacity = 0
    leaves.count = 0
    leaves.elements = NULL
    resize_c(&leaves, 10)
    cdef SIZE_t i
    for i in range(10):
        leaves.elements[i] = create_int_array(i*2 + 1)
    leaves.count = 10
    return leaves

cpdef void _test_copy_int_array():
    cdef IntArray to_copy = create_int_array(4)
    cdef IntArray copied = copy_int_array(&to_copy)
    assert copied.count == copied.capacity == to_copy.count
    assert copied.capacity <= to_copy.capacity
    for i in range(10):
        assert copied.elements[i] == to_copy.elements[i]
    to_copy.elements[1] = 0
    assert to_copy.elements[1] != copied.elements[1]
    free(to_copy.elements)
    free(copied.elements)

cpdef void _test_copy_leaves():
    cdef Leaves to_copy = create_leaves()
    cdef Leaves copied = copy_leaves(&to_copy)
    assert copied.count == copied.capacity == to_copy.count
    assert copied.capacity <= to_copy.capacity
    for i in range(10):
        assert copied.elements[i].count == copied.elements[i].capacity == to_copy.elements[i].count
        assert copied.elements[i].capacity <= to_copy.elements[i].capacity
        for j in range(10):
            assert copied.elements[i].elements[j] == to_copy.elements[i].elements[j]
    to_copy.elements[1].elements[1] = 0
    assert to_copy.elements[1].elements[1] != copied.elements[1].elements[1]
    for i in range(10):
        free(to_copy.elements[i].elements)
        free(copied.elements[i].elements)
    free(to_copy.elements)
    free(copied.elements)

# =============================================================================
# Stack data structure - copied from sklearn.tree._utils
# but changed to contain relevant information
# =============================================================================

cdef class Stack:
    """A LIFO data structure.
    Attributes
    ----------
    capacity : SIZE_t
        The elements the stack can hold; if more added then ``self.stack_``
        needs to be resized.
    top : SIZE_t
        The number of elements currently on the stack.
    stack : StackRecord pointer
        The stack of records (upward in the stack corresponds to the right).
    """

    def __cinit__(self, SIZE_t capacity):
        self.capacity = capacity
        self.top = 0
        self.stack_ = <StackRecord*> malloc(capacity * sizeof(StackRecord))

    def __dealloc__(self):
        free(self.stack_)

    cdef bint is_empty(self) nogil:
        return self.top <= 0

    cdef int push(self, SIZE_t new_parent_id, SIZE_t old_self_id,
                  bint is_left,
                  SIZE_t feature, double threshold, SIZE_t depth,
                  ) nogil except -1:
        """Push a new element onto the stack.
        Return -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        cdef SIZE_t top = self.top
        cdef StackRecord* stack = NULL

        # Resize if capacity not sufficient
        if top >= self.capacity:
            self.capacity *= 2
            # Since safe_realloc can raise MemoryError, use `except -1`
            safe_realloc(&self.stack_, self.capacity)

        stack = self.stack_
        stack[top].new_parent_id = new_parent_id
        stack[top].old_self_id = old_self_id
        stack[top].is_left = is_left
        stack[top].feature = feature
        stack[top].threshold = threshold
        stack[top].depth = depth

        # Increment stack pointer
        self.top = top + 1
        return 0

    cdef int pop(self, StackRecord* res) nogil:
        """Remove the top element from the stack and copy to ``res``.
        Returns 0 if pop was successful (and ``res`` is set); -1
        otherwise.
        """
        cdef SIZE_t top = self.top
        cdef StackRecord* stack = self.stack_

        if top <= 0:
            return -1

        res[0] = stack[top - 1]
        self.top = top - 1

        return 0