# copied from sklearn.tree._utils.pxd
# Authors: Gilles Louppe <g.louppe@gmail.com>
#          Peter Prettenhofer <peter.prettenhofer@gmail.com>
#          Arnaud Joly <arnaud.v.joly@gmail.com>
#          Jacob Schreiber <jmschreiber91@gmail.com>
#          Nelson Liu <nelson@nelsonliu.me>
#
# License: BSD 3 clause

# See _utils.pyx for details.


import numpy as np
cimport numpy as np
from tree.tree cimport Node

ctypedef np.npy_float64 DOUBLE_t         # Type of y, sample_weight
ctypedef np.npy_intp SIZE_t              # Type for indices and counters

ctypedef struct Leaves:
    SIZE_t count                # current number of elements in DynamicArray
    SIZE_t capacity             # current max capacity of DynamicArray
    IntArray* elements          # pointer to Array with elements

ctypedef struct IntArray:
    SIZE_t count                # current number of elements in DynamicArray
    SIZE_t capacity             # current max capacity of DynamicArray
    SIZE_t* elements            # pointer to Array with elements

ctypedef fused realloc_ptr:
    # Add pointer types here as needed.
    (SIZE_t*)
    (DOUBLE_t*)
    (DOUBLE_t**)
    (Node*)
    (StackRecord*)
    (Leaves*)
    (IntArray*)

ctypedef fused DynamicArray:
    Leaves
    IntArray

cdef realloc_ptr safe_realloc(realloc_ptr* p, size_t nelems) nogil except *

cdef int resize(DynamicArray* array, SIZE_t capacity) nogil except -1
cdef int resize_c(DynamicArray* array, SIZE_t capacity=*) nogil except -1

cdef np.ndarray sizet_ptr_to_ndarray(SIZE_t* data, SIZE_t size)

cdef copy_int_array(IntArray* old_array, IntArray* new_array)
cdef copy_leaves(Leaves* old_leaves, Leaves* new_leaves)

# =============================================================================
# Stack data structure - copied from sklearn.tree._utils
# but changed to contain relevant information
# =============================================================================

# A record on the stack for depth-first tree growing
cdef struct StackRecord:
    SIZE_t new_parent_id
    SIZE_t old_self_id
    bint is_left
    SIZE_t feature
    double threshold
    SIZE_t depth

cdef class Stack:
    cdef SIZE_t capacity
    cdef SIZE_t top
    cdef StackRecord* stack_

    cdef bint is_empty(self) nogil
    cdef int push(self, SIZE_t new_parent_id, SIZE_t old_self_id,
                  bint is_left,
                  SIZE_t feature, double threshold, SIZE_t depth,
                  ) nogil except -1
    cdef int pop(self, StackRecord* res) nogil