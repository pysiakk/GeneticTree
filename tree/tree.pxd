# distutils: define_macros=CYTHON_TRACE_NOGIL=1

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

ctypedef np.npy_float64 DOUBLE_t        # Type of y
ctypedef np.npy_float32 DTYPE_t         # Type of X
ctypedef np.npy_intp SIZE_t             # Type for indices and counters

cdef struct Node:
    # Base storage structure for the nodes in a Tree object
    SIZE_t left_child                   # id of the left child of the node
    SIZE_t right_child                  # id of the right child of the node
    SIZE_t parent                       # id of the parent of the node
    SIZE_t feature                      # Feature used for splitting the node
                                        # or predicted class if the node is leaf
    DOUBLE_t threshold                  # Threshold value at the node
    SIZE_t depth                        # the size of path from root to node


cdef class Observation:
    # Base storage structure of observation metadata
    cdef public SIZE_t proper_class     # the class of observation in y
    cdef public SIZE_t current_class    # the class of current node
    cdef public SIZE_t observation_id   # id of observation
    cdef public SIZE_t last_node_id     # node id that observation must be below
                                        # usually the last node_id before mutation or crossover


cdef class Tree:
    # The Tree object is a binary tree structure.

    # Input/Output layout
    cdef public SIZE_t n_features       # Number of features in X
    cdef public SIZE_t n_classes        # Number of classes in y
    cdef public SIZE_t n_thresholds     # Number of possible thresholds to mutate between

    # Inner structures: values are stored separately from node structure,
    # since size is determined at runtime.
    cdef public SIZE_t depth            # Max depth seen of the tree
    cdef public SIZE_t node_count       # Counter for node IDs
    cdef public SIZE_t capacity         # Capacity of tree, in terms of nodes
    cdef Node* nodes                    # Array of nodes

    cdef public dict observations           # Dictionary with y array metadata
    cdef public DTYPE_t[:, :] thresholds    # Array with possible thresholds for each feature

    # Methods
    cdef _resize_by_initial_depth(self, int initial_depth)
    cdef int _resize(self, SIZE_t capacity) nogil except -1
    cdef int _resize_c(self, SIZE_t capacity=*) nogil except -1

    cdef SIZE_t _add_node(self, SIZE_t parent, bint is_left, bint is_leaf,
                          SIZE_t feature, double threshold, SIZE_t depth,
                          SIZE_t class_number) nogil except -1
    cdef np.ndarray _get_node_ndarray(self)

    # Mutation functions
    cpdef mutate_random_node(self)
    cpdef mutate_random_class_or_threshold(self)
    cpdef mutate_random_feature(self)
    cpdef mutate_random_threshold(self)
    cpdef mutate_random_class(self)

    cdef _mutate_feature(self, SIZE_t node_id)
    cdef _mutate_threshold(self, SIZE_t node_id, bint feature_changed)
    cdef _mutate_class(self, SIZE_t node_id)

    cdef public SIZE_t get_random_node(self)
    cdef SIZE_t _get_random_decision_node(self)
    cdef SIZE_t _get_random_leaf(self)

    cdef SIZE_t _get_new_random_feature(self, SIZE_t last_feature)
    cdef DOUBLE_t _get_new_random_threshold(self, DOUBLE_t last_threshold, SIZE_t feature, bint feature_changed)
    cdef SIZE_t _get_new_random_class(self, SIZE_t last_class)

    cdef _change_feature_or_class(self, SIZE_t node_id, SIZE_t new_feature)
    cdef _change_threshold(self, SIZE_t node_id, DOUBLE_t new_threshold)

    # Observations functions
    cpdef initialize_observations(self, object X, np.ndarray y)
    cpdef assign_all_not_registered_observations(self, object X)

    cdef _assign_leaf_for_observation(self, Observation observation, SIZE_t node_id)
    cdef SIZE_t _find_leaf_for_observation(self, SIZE_t observation_id, DTYPE_t[:, :] X_ndarray,
                                        SIZE_t node_id_to_start) nogil

    cdef _remove_observations_below_node(self, SIZE_t node_id)
    cdef _remove_observations_of_node_recurrent(self, SIZE_t current_node_id, SIZE_t node_id_as_last)
    cdef _remove_observations_of_node(self, SIZE_t current_node_id, SIZE_t node_id_as_last)

    # Prediction functions
    cdef prepare_tree_to_prediction(self)
    cpdef np.ndarray predict(self, object X)

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
