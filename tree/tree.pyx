# cython: linetrace=True

# code copied from https://github.com/scikit-learn/scikit-learn/blob/fd237278e895b42abe8d8d09105cbb82dc2cbba7/sklearn/tree/_tree.pyx
# notes above this file:

# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# Authors: Gilles Louppe <g.louppe@gmail.com>
#          Peter Prettenhofer <peter.prettenhofer@gmail.com>
#          Brian Holt <bdholt1@gmail.com>
#          Noel Dawe <noel@dawe.me>
#          Satrajit Gosh <satrajit.ghosh@gmail.com>
#          Lars Buitinck
#          Arnaud Joly <arnaud.v.joly@gmail.com>
#          Joel Nothman <joel.nothman@gmail.com>
#          Fares Hedayati <fares.hedayati@gmail.com>
#          Jacob Schreiber <jmschreiber91@gmail.com>
#          Nelson Liu <nelson@nelsonliu.me>
#
# License: BSD 3 clause

# from the scikit-learn was copied general Node struct, but in this project
# the Node fields was slightly changed
# also was copied a Tree as a class also with some changes to use in this project
# the last thing copied was utils (exactly 2 functions) used directly in tree

from cpython cimport Py_INCREF, PyObject, PyTypeObject
from cpython.pycapsule cimport PyCapsule_IsValid, PyCapsule_GetPointer

from libc.stdlib cimport free
from libc.string cimport memcpy
from libc.stdint cimport SIZE_MAX

from tree._utils cimport safe_realloc

from tree.observations cimport Observations
from tree.observations import Observations, copy_observations

import numpy as np
cimport numpy as np
np.import_array()

from scipy.sparse import dok_matrix, csr_matrix

from numpy import float64 as DOUBLE

cdef extern from "numpy/arrayobject.h":
    object PyArray_NewFromDescr(PyTypeObject* subtype, np.dtype descr,
                                int nd, np.npy_intp* dims,
                                np.npy_intp* strides,
                                void* data, int flags, object obj)

# =============================================================================
# Types and constants
# =============================================================================

cdef DTYPE_t EPSILON = 0.00001
TREE_LEAF = -1
TREE_UNDEFINED = -2
NODE_REMOVED = -3
cdef SIZE_t _TREE_LEAF = TREE_LEAF
cdef SIZE_t _TREE_UNDEFINED = TREE_UNDEFINED
cdef SIZE_t _NODE_REMOVED = NODE_REMOVED

# Repeat struct definition for numpy
NODE_DTYPE = np.dtype({
    'names': ['left_child', 'right_child', 'parent', 'feature', 'threshold', 'depth'],
    'formats': [np.intp, np.intp, np.intp, np.intp, np.float64, np.intp],
    'offsets': [
        <Py_ssize_t> &(<Node*> NULL).left_child,
        <Py_ssize_t> &(<Node*> NULL).right_child,
        <Py_ssize_t> &(<Node*> NULL).parent,
        <Py_ssize_t> &(<Node*> NULL).feature,
        <Py_ssize_t> &(<Node*> NULL).threshold,
        <Py_ssize_t> &(<Node*> NULL).depth
    ]
})

cdef class Tree:
    property children_left:
        def __get__(self):
            return self._get_node_ndarray()['left_child'][:self.nodes.count]

    property children_right:
        def __get__(self):
            return self._get_node_ndarray()['right_child'][:self.nodes.count]

    property parent:
        def __get__(self):
            return self._get_node_ndarray()['parent'][:self.nodes.count]

    property n_leaves:
        def __get__(self):
            return np.sum(np.logical_and(
                self.children_left == -1,
                self.children_right == -1))

    property feature:
        def __get__(self):
            return self._get_node_ndarray()['feature'][:self.nodes.count]

    property threshold:
        def __get__(self):
            return self._get_node_ndarray()['threshold'][:self.nodes.count]

    property nodes_depth:
        def __get__(self):
            return self._get_node_ndarray()['depth'][:self.nodes.count]

    property node_count:
        def __get__(self):
            return self.nodes.count

    property proper_classified:
        def __get__(self):
            return self.observations.proper_classified

    def __cinit__(self, int n_classes,
                  object X,
                  SIZE_t[:] y,
                  DTYPE_t[:] weights,
                  DTYPE_t[:, :] thresholds,
                  uint64_t seed):
        """Constructor."""
        self.n_features = X.shape[1]
        self.n_observations = X.shape[0]
        self.n_classes = n_classes
        self.n_thresholds = thresholds.shape[0]

        self.X = X
        self.y = y
        self.weights = weights
        self.thresholds = thresholds

        # Inner structures
        self.depth = 0

        self.nodes = NULL
        safe_realloc(&self.nodes, 1)
        self.nodes.count = 0
        self.nodes.capacity = 0
        self.nodes.elements = NULL

        self.removed_nodes = NULL
        safe_realloc(&self.removed_nodes, 1)
        self.removed_nodes.count = 0
        self.removed_nodes.capacity = 0
        self.removed_nodes.elements = NULL

        self.observations = Observations(X, y, weights)

        self.seed1 = seed
        self.seed2 = 987654321
        self.seed3 = 43219876
        self.seed4 = 6543217

    def __dealloc__(self):
        """Destructor."""
        # Free all inner structures
        free(self.nodes.elements)
        free(self.nodes)
        free(self.removed_nodes.elements)
        free(self.removed_nodes)

    def __reduce__(self):
        """Reduce re-implementation, for pickling."""
        # never pickle trees during fit
        # after unpickling need to pass pointers to X, y and thresholds arrays
        empty_2d_array = np.empty((1, 1), dtype=np.float32)
        empty_1d_array_int = np.empty(1, dtype=np.intp)
        empty_1d_array = np.empty(1, dtype=np.float32)
        return (Tree,
               (self.n_classes,
               empty_2d_array,
               empty_1d_array_int,
               empty_1d_array,
               empty_2d_array,
               self.seed1,
               ), self.__getstate__())

    def __getstate__(self):
        """Getstate re-implementation, for pickling."""
        # TODO pickle observations

        state = {}
        # capacity is inferred during the __setstate__ using nodes
        state["depth"] = self.depth
        state["node_count"] = self.nodes.count
        state["nodes"] = self._get_node_ndarray()

        state["n_features"] = self.n_features
        state["n_observations"] = self.n_observations
        state["n_thresholds"] = self.n_thresholds
        state["seed2"] = self.seed2
        state["seed3"] = self.seed3
        state["seed4"] = self.seed4

        return state

    def __setstate__(self, state):
        """Setstate re-implementation, for unpickling."""
        # TODO unpickle observations
        self.depth = state["depth"]
        self.nodes.count = state["node_count"]

        self.n_features = state["n_features"]
        self.n_observations = state["n_observations"]
        self.n_thresholds = state["n_thresholds"]
        self.seed2 = state["seed2"]
        self.seed3 = state["seed3"]
        self.seed4 = state["seed4"]

        if 'nodes' not in state:
            raise ValueError('You have loaded Tree version which '
                             'cannot be imported')

        node_ndarray = state['nodes']

        if (node_ndarray.ndim != 1 or
                node_ndarray.dtype != NODE_DTYPE or
                not node_ndarray.flags.c_contiguous):
            raise ValueError('Did not recognise loaded array layout')

        self.unpickle_nodes(node_ndarray)

    def unpickle_nodes(self, node_ndarray):
        self.nodes.capacity = node_ndarray.shape[0]
        if resize_c(self.nodes, self.nodes.capacity) != 0:
            raise MemoryError("resizing tree to %d" % self.nodes.capacity)
        nodes = memcpy(self.nodes.elements, (<np.ndarray> node_ndarray).data,
                       self.nodes.capacity * sizeof(Node))

    cpdef resize_by_initial_depth(self, int initial_depth):
        if initial_depth <= 10:
            init_capacity = (2 ** (initial_depth + 1)) - 1
        else:
            init_capacity = 2047

        resize(self.nodes, init_capacity)

    cdef SIZE_t _add_node(self, SIZE_t parent, bint is_left, bint is_leaf,
                          SIZE_t feature, double threshold, SIZE_t depth,
                          SIZE_t class_number) nogil except -1:
        """Add a node to the tree.
        The new node registers itself as the child of its parent.
        Returns (size_t)(-1) on error.
        """
        cdef SIZE_t node_id = self.nodes.count

        if self.removed_nodes.count != 0:
            self.removed_nodes.count -= 1
            node_id = self.removed_nodes.elements[self.removed_nodes.count]
            self.nodes.count -= 1  # because it will be added 1 at the end

        if node_id >= self.nodes.capacity:
            if resize_c(self.nodes) != 0:
                return SIZE_MAX

        cdef Node* node = &self.nodes.elements[node_id]

        node.parent = parent
        node.depth = depth

        if parent != _TREE_UNDEFINED:
            if is_left:
                self.nodes.elements[parent].left_child = node_id
            else:
                self.nodes.elements[parent].right_child = node_id

        if is_leaf:
            node.left_child = _TREE_LEAF
            node.right_child = _TREE_LEAF
            node.feature = class_number             # class instead of feature
            node.threshold = _TREE_UNDEFINED

        else:
            # left_child and right_child will be set later
            node.feature = feature
            node.threshold = threshold

        self.nodes.count += 1

        return node_id

    cdef SIZE_t mark_nodes_as_removed(self, SIZE_t below_node_id):
        if self.removed_nodes.count >= self.removed_nodes.capacity:
            if resize_c(self.removed_nodes) != 0:
                return SIZE_MAX

        self.removed_nodes.elements[self.removed_nodes.count] = below_node_id
        self.nodes.elements[below_node_id].parent = _NODE_REMOVED

        self.removed_nodes.count += 1

        if self.nodes.elements[below_node_id].left_child != _TREE_LEAF:
            self.mark_nodes_as_removed(self.nodes.elements[below_node_id].left_child)
            self.mark_nodes_as_removed(self.nodes.elements[below_node_id].right_child)

    cdef SIZE_t compact_removed_nodes(self, SIZE_t crossover_point) nogil:
        cdef SIZE_t i
        cdef SIZE_t* node_id = self.removed_nodes.elements
        cdef SIZE_t copy_from
        with nogil:
            for i in range(self.removed_nodes.count):
                if i != 0:
                    node_id += 1
                if self.nodes.count <= node_id[0]:
                    continue
                copy_from = self.nodes.count - 1
                while self.nodes.elements[copy_from].parent == _NODE_REMOVED:
                    copy_from -= 1
                self.nodes.count = copy_from
                if node_id[0] >= copy_from:
                    self.nodes.count += 1
                    continue
                if copy_from == crossover_point:
                    crossover_point = node_id[0]
                self._copy_node(&self.nodes.elements[copy_from], copy_from, &self.nodes.elements[node_id[0]], node_id[0])

            self.removed_nodes.count = 0
            self.removed_nodes.capacity = 0
            free(self.removed_nodes.elements)
            self.removed_nodes.elements = NULL
        resize_c(self.nodes, self.nodes.count)
        return crossover_point

    cdef void _copy_node(self, Node* from_node, SIZE_t from_node_id, Node* to_node, SIZE_t to_node_id) nogil:
        to_node.depth = from_node.depth
        to_node.threshold = from_node.threshold
        to_node.feature = from_node.feature
        to_node.parent = from_node.parent
        if self.nodes.elements[from_node.parent].left_child == from_node_id:
            self.nodes.elements[from_node.parent].left_child = to_node_id
        else:
            self.nodes.elements[from_node.parent].right_child = to_node_id
        to_node.left_child = from_node.left_child
        to_node.right_child = from_node.right_child
        if from_node.left_child != _TREE_LEAF:
            self.nodes.elements[from_node.left_child].parent = to_node_id
            self.nodes.elements[from_node.right_child].parent = to_node_id

    cdef np.ndarray _get_node_ndarray(self):
        """Wraps nodes as a NumPy struct array.
        The array keeps a reference to this Tree, which manages the underlying
        memory. Individual fields are publicly accessible as properties of the
        Tree.
        """
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> self.nodes.count
        cdef np.npy_intp strides[1]
        strides[0] = sizeof(Node)
        cdef np.ndarray arr
        Py_INCREF(NODE_DTYPE)
        arr = PyArray_NewFromDescr(<PyTypeObject *> np.ndarray,
                                   <np.dtype> NODE_DTYPE, 1, shape,
                                   strides, <void*> self.nodes.elements,
                                   np.NPY_DEFAULT, None)
        Py_INCREF(self)
        arr.base = <PyObject*> self
        return arr

    cdef change_feature_or_class(self, SIZE_t node_id, SIZE_t new_feature):
        self.nodes.elements[node_id].feature = new_feature

    cdef change_threshold(self, SIZE_t node_id, DOUBLE_t new_threshold):
        self.nodes.elements[node_id].threshold = new_threshold

# ===========================================================================================================
# Random functions
# ===========================================================================================================

    # it is one of JKISS generators - found in lecture notes
    # KISS in JKISS means keep it simple stupid
    cdef SIZE_t randint(self, SIZE_t lb, SIZE_t ub) nogil:
        cdef uint64_t temp
        cdef SIZE_t result
        self.seed1 = 314527869 * self.seed1 + 1234567
        self.seed2 ^= self.seed2 << 5
        self.seed2 ^= self.seed2 >> 7
        self.seed2 ^= self.seed2 << 22
        temp = 4294584393ULL * self.seed3 + self.seed4
        self.seed4 = temp >> 32
        self.seed3 = temp
        result = lb + (self.seed1 + self.seed2 + self.seed3) % (ub - lb)
        return result

    cpdef public SIZE_t get_random_node(self):
        cdef SIZE_t random_id = self.randint(0, self.nodes.count)
        return random_id

    cdef SIZE_t get_random_decision_node(self):
        cdef SIZE_t random_id = self.randint(0, self.nodes.count)
        while self.nodes.elements[random_id].left_child == _TREE_LEAF:
            random_id = self.randint(0, self.nodes.count)
        return random_id

    cdef SIZE_t get_random_leaf(self):
        cdef SIZE_t random_id = self.randint(0, self.nodes.count)
        while self.nodes.elements[random_id].left_child != _TREE_LEAF:
            random_id = self.randint(0, self.nodes.count)
        return random_id

    cdef SIZE_t get_new_random_feature(self, SIZE_t last_feature):
        cdef SIZE_t new_feature = self.randint(0, self.n_features - 1)
        if new_feature >= last_feature:
            new_feature += 1
        return new_feature

    cdef DOUBLE_t get_new_random_threshold(self, DOUBLE_t last_threshold, SIZE_t feature, bint feature_changed):
        cdef SIZE_t new_threshold_index
        if feature_changed == 1:
            new_threshold_index = self.randint(0, self.n_thresholds)
        else:
            new_threshold_index = self.randint(0, self.n_thresholds - 1)
            if self.thresholds[new_threshold_index, feature] >= last_threshold:
                new_threshold_index += 1
        return self.thresholds[new_threshold_index, feature]

    cdef SIZE_t get_new_random_class(self, SIZE_t last_class):
        cdef SIZE_t new_class = self.randint(0, self.n_classes - 1)
        if new_class >= last_class:
            new_class += 1
        return new_class

# ===========================================================================================================
# Observations functions
# ===========================================================================================================
    # initialization of observations
    cpdef initialize_observations(self):
        self.observations.initialize_observations(self)

    # finding proper leaf for observation
    cdef SIZE_t _find_leaf_for_observation(self, SIZE_t observation_id, DTYPE_t[:, :] X_ndarray,
                                               SIZE_t node_id_to_start) nogil:
        cdef DTYPE_t[:] X_row = X_ndarray[observation_id, :]
        cdef SIZE_t current_node_id = node_id_to_start
        cdef SIZE_t feature
        cdef DOUBLE_t threshold
        with nogil:
            while self.nodes.elements[current_node_id].left_child != _TREE_LEAF:
                feature = self.nodes.elements[current_node_id].feature
                threshold = self.nodes.elements[current_node_id].threshold
                if X_row[feature] <= threshold:
                    current_node_id = self.nodes.elements[current_node_id].left_child
                else:
                    current_node_id = self.nodes.elements[current_node_id].right_child
        return current_node_id

# ===========================================================================================================
# Prediction functions
# ===========================================================================================================

    cpdef prepare_tree_to_prediction(self):
        cdef DTYPE_t[:] observations_in_class
        cdef IntArray observations
        cdef SIZE_t node_id
        cdef SIZE_t i
        self.probabilities = dok_matrix((self.nodes.count, self.n_classes), dtype=np.float32)
        # for each node (f the node is leaf) change class for the most occurring
        for node_id in range(self.nodes.count):
            # if it is leaf and has one or more observation
            if self.nodes.elements[node_id].left_child == _TREE_LEAF:
                if self.nodes.elements[node_id].right_child != _TREE_LEAF:
                    observations_in_class = np.zeros(self.n_classes, dtype=np.float32)
                    observations = self.observations.leaves.elements[self.nodes.elements[node_id].right_child]
                    for i in range(observations.count):
                        observations_in_class[self.y[observations.elements[i]]] += 1
                    # change class if it is not the maximum value
                    if observations_in_class[self.nodes.elements[node_id].feature] != np.max(observations_in_class):
                        self.nodes.elements[node_id].feature = np.argmax(observations_in_class)
                else:
                    observations_in_class = np.ones(self.n_classes, dtype=np.float32)

                # add probabilities of leaf
                observations_in_class[self.nodes.elements[node_id].feature] += EPSILON
                self.probabilities[node_id, :] = observations_in_class / np.sum(observations_in_class)

        self.probabilities = csr_matrix(self.probabilities)

    cpdef void remove_variables(self):
        self.observations = None
        self.X = None
        self.y = None
        self.weights = None
        self.thresholds = None

    cpdef np.ndarray predict(self, object X):
        cdef DTYPE_t[:, :] X_ndarray = X
        cdef n_observations = X_ndarray.shape[0]
        cdef np.ndarray y = np.empty(n_observations, dtype=np.intp)
        cdef SIZE_t observation_id

        for observation_id in range(n_observations):
            node_id = self._find_leaf_for_observation(observation_id, X_ndarray, 0)
            y[observation_id] = self.nodes.elements[node_id].feature  # feature means class for leaf

        return y

    cpdef object predict_proba(self, object X):
        cdef DTYPE_t[:, :] X_ndarray = X
        cdef n_observations = X_ndarray.shape[0]
        y_prob = dok_matrix((n_observations, self.n_classes), dtype=np.float32)
        cdef SIZE_t observation_id

        for observation_id in range(n_observations):
            node_id = self._find_leaf_for_observation(observation_id, X_ndarray, 0)
            y_prob[observation_id, :] = self.probabilities[node_id, :][0]

        return csr_matrix(y_prob)

    cpdef np.ndarray apply(self, object X):
        cdef DTYPE_t[:, :] X_ndarray = X
        cdef n_observations = X_ndarray.shape[0]
        cdef np.ndarray nodes = np.empty(n_observations, dtype=np.intp)
        cdef SIZE_t observation_id

        for observation_id in range(n_observations):
            nodes[observation_id] = self._find_leaf_for_observation(observation_id, X_ndarray, 0)

        return nodes


cpdef Tree copy_tree(Tree tree):
    cdef Tree tree_copied = Tree(tree.n_classes, tree.X, tree.y, tree.weights, tree.thresholds, np.random.randint(10**8))
    tree_copied.depth = tree.depth
    tree_copied.nodes.count = tree.nodes.count

    tree_copied.unpickle_nodes(tree._get_node_ndarray())
    tree_copied.observations = copy_observations(tree.observations)

    return tree_copied

cpdef void test_independence_of_copied_tree(Tree tree):
    cdef Tree tree_copied = copy_tree(tree)

    tree_copied.nodes.elements[0].parent += 1
    assert tree.nodes.elements[0].parent != tree_copied.nodes.elements[0].parent
