# code copied from https://github.com/scikit-learn/scikit-learn/blob/fd237278e895b42abe8d8d09105cbb82dc2cbba7/sklearn/tree/_tree.pxd
# notes above this file:

# Authors: Gilles Louppe <g.louppe@gmail.com>
#          Peter Prettenhofer <peter.prettenhofer@gmail.com>
#          Brian Holt <bdholt1@gmail.com>
#          Joel Nothman <joel.nothman@gmail.com>
#          Arnaud Joly <arnaud.v.joly@gmail.com>
#          Jacob Schreiber <jmschreiber91@gmail.com>
#          Nelson Liu <nelson@nelsonliu.me>
#
# License: BSD 3 clause

# See _tree.pyx for details.

import numpy as np
cimport numpy as np

ctypedef np.npy_float64 DOUBLE_t         # Type of y, sample_weight
ctypedef np.npy_intp SIZE_t              # Type for indices and counters

cdef struct Node:
    # Base storage structure for the nodes in a Tree object
    SIZE_t left_child                    # id of the left child of the node
    SIZE_t right_child                   # id of the right child of the node
    SIZE_t parent                        # id of the parent of the node
    SIZE_t feature                       # Feature used for splitting the node
    DOUBLE_t threshold                   # Threshold value at the node
    SIZE_t depth                         # the size of path from root to node


cdef struct Observation:
    # Base storage structure of observation metadata
    SIZE_t proper_class                 # the class of observation in y
    SIZE_t current_class                # the class of current node
    SIZE_t observation_id               # id of observation
    SIZE_t last_node_id                 # node id that observation must be below
                                        # usually the last node_id before mutation or crossover


cdef class Tree:
    # The Tree object is a binary tree structure.

    # Input/Output layout
    cdef public SIZE_t n_features        # Number of features in X
    cdef public SIZE_t n_classes         # Number of classes in y
    cdef public SIZE_t n_outputs         # Number of outputs in y

    # Inner structures: values are stored separately from node structure,
    # since size is determined at runtime.
    cdef public SIZE_t max_depth         # Max depth of the tree
    cdef public SIZE_t node_count        # Counter for node IDs
    cdef public SIZE_t capacity          # Capacity of tree, in terms of nodes
    cdef Node* nodes                     # Array of nodes
    cdef double* value                   # (capacity, 1, n_classes) array of values
    cdef SIZE_t value_stride             # = 1 * n_classes
    # TODO PoC of dictionary structure
    # TODO create dictionary during initialization of trees
    # TODO update dictionary during mutation
    # TODO create dictionary for new trees during crossing
    cdef dict observations

    # Methods
    cdef SIZE_t _add_node(self, SIZE_t parent, bint is_left, bint is_leaf,
                          SIZE_t feature, double threshold, SIZE_t depth,
                          SIZE_t class_number) nogil except -1
    cdef int _resize(self, SIZE_t capacity) nogil except -1
    cdef int _resize_c(self, SIZE_t capacity=*) nogil except -1

    cdef np.ndarray _get_value_ndarray(self)
    cdef np.ndarray _get_node_ndarray(self)

    # Mutation functions
    cpdef mutate_random_node(self)
    cpdef mutate_random_feature(self)
    cpdef mutate_random_threshold(self)
    cpdef mutate_random_class(self)

    cdef _mutate_feature(self, SIZE_t node_id)
    cdef _mutate_threshold(self, SIZE_t node_id)
    cdef _mutate_class(self, SIZE_t node_id)

    cdef SIZE_t _get_random_node(self)
    cdef SIZE_t _get_random_decision_node(self)
    cdef SIZE_t _get_random_leaf(self)

    cdef SIZE_t _get_random_feature(self)
    cdef DOUBLE_t _get_random_threshold(self)
    cdef SIZE_t _get_random_class(self)

    cdef _change_feature_or_class(self, SIZE_t node_id, SIZE_t new_feature)
    cdef _change_threshold(self, SIZE_t node_id, DOUBLE_t new_threshold)

    # Observations functions
    cdef _remove_observations_below_node(self, SIZE_t node_id)
    cdef _remove_observations_of_node_recurrent(self, SIZE_t current_node_id, SIZE_t node_id_as_last)
    cdef _remove_observations_of_node(self, SIZE_t current_node_id, SIZE_t node_id_as_last)

    # commented out functions
    # cpdef np.ndarray predict(self, object X)

    # cpdef np.ndarray apply(self, object X)
    # cdef np.ndarray _apply_dense(self, object X)
    # cdef np.ndarray _apply_sparse_csr(self, object X)

    # cpdef object decision_path(self, object X)
    # cdef object _decision_path_dense(self, object X)
    # cdef object _decision_path_sparse_csr(self, object X)

    # cpdef compute_feature_importances(self, normalize=*)

    # Multithreading test functions
    cpdef test_function_with_args_core(self, char* name, long long size, int print_size)
    cpdef test_function_with_args(self, char* name, long long size, int print_size)

    cpdef time_test2(self, long long size)
    cpdef time_test(self, long long size)

    cpdef time_test_nogil(self, long long size)
    cdef void _time_test_nogil_(self, long long size) nogil
