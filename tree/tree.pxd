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

from tree.observations cimport Observations
from tree.observations import Observations
from tree._utils cimport Node, NodeArray, IntArray, resize, resize_c

ctypedef np.npy_float64 DOUBLE_t        # Type of thresholds
ctypedef np.npy_float32 DTYPE_t         # Type of X
ctypedef np.npy_intp SIZE_t             # Type for indices and counters
ctypedef np.npy_uint64 uint64_t             # Type for random generator JKISS

cdef class Tree:
    # The Tree object is a binary tree structure.

    # Sizes of arrays
    cdef public SIZE_t n_features       # Number of features in X
    cdef public SIZE_t n_observations   # Number of observations in X and y
    cdef public SIZE_t n_classes        # Number of classes in y
    cdef public SIZE_t n_thresholds     # Number of possible thresholds to mutate between

    # Inner structures: values are stored separately from node structure,
    # since size is determined at runtime.
    cdef public SIZE_t depth            # Max depth seen of the tree
    cdef NodeArray* nodes               # Array of nodes
    cdef IntArray* removed_nodes

    cdef Observations observations      # Class with y array metadata
    cdef public object probabilities    # Probabilities of classes in nodes

    cdef public DTYPE_t[:, :] thresholds    # Array with possible thresholds for each feature
    cdef public object X                    # Array with observations features (TODO: possibility of sparse array)
    cdef public SIZE_t[:] y                 # Array with classes of observations
    cdef public DTYPE_t[:] sample_weight          # Array with sample_weight of observations

    cdef uint64_t seed1
    cdef uint64_t seed2
    cdef uint64_t seed3
    cdef uint64_t seed4

    # Methods
    cpdef resize_by_initial_depth(self, int initial_depth)

    cdef SIZE_t _add_node(self, SIZE_t parent, bint is_left, bint is_leaf,
                          SIZE_t feature, double threshold, SIZE_t depth,
                          SIZE_t class_number) nogil except -1

    cdef SIZE_t mark_nodes_as_removed(self, SIZE_t node_id)
    cdef SIZE_t compact_removed_nodes(self, SIZE_t crossover_point) nogil
    cdef void _copy_node(self, Node* from_node, SIZE_t from_node_id, Node* to_node, SIZE_t to_node_id) nogil

    cdef np.ndarray _get_node_ndarray(self)

    cdef change_feature_or_class(self, SIZE_t node_id, SIZE_t new_feature)
    cdef change_threshold(self, SIZE_t node_id, DOUBLE_t new_threshold)

    # Random functions
    cdef SIZE_t randint_c(self, SIZE_t lb, SIZE_t ub) nogil
    cpdef public SIZE_t get_random_node(self)
    cdef SIZE_t get_random_decision_node(self)
    cdef SIZE_t get_random_leaf(self)
    cdef SIZE_t get_new_random_feature(self, SIZE_t last_feature)
    cdef DOUBLE_t get_new_random_threshold(self, DOUBLE_t last_threshold, SIZE_t feature, bint feature_changed)
    cdef SIZE_t get_new_random_class(self, SIZE_t last_class)

    # Observations functions
    cpdef initialize_observations(self)

    cdef SIZE_t _find_leaf_for_observation(self, SIZE_t observation_id, DTYPE_t[:, :] X_ndarray,
                                        SIZE_t node_id_to_start) nogil

    # Prediction functions
    cpdef prepare_tree_to_prediction(self)
    cpdef void remove_variables(self)
    cpdef np.ndarray predict(self, object X)
    cpdef object predict_proba(self, object X)
    cpdef np.ndarray apply(self, object X)


cpdef Tree copy_tree(Tree tree, bint same_seed=*)
