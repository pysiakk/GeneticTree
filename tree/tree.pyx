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

from libc.stdlib cimport free
from libc.string cimport memcpy
from libc.stdint cimport SIZE_MAX
from libc.stdio cimport printf

from tree._utils cimport safe_realloc

import multiprocessing

import numpy as np
cimport numpy as np
np.import_array()

from numpy import float32 as DTYPE
from numpy import float64 as DOUBLE

cdef extern from "numpy/arrayobject.h":
    object PyArray_NewFromDescr(PyTypeObject* subtype, np.dtype descr,
                                int nd, np.npy_intp* dims,
                                np.npy_intp* strides,
                                void* data, int flags, object obj)

# =============================================================================
# Types and constants
# =============================================================================

TREE_LEAF = -1
TREE_UNDEFINED = -2
NOT_REGISTERED = -1
cdef SIZE_t _TREE_LEAF = TREE_LEAF
cdef SIZE_t _TREE_UNDEFINED = TREE_UNDEFINED
cdef SIZE_t _NOT_REGISTERED = NOT_REGISTERED

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

cdef class Observation:
    def __cinit__(self, SIZE_t proper_class, SIZE_t current_class,
                  SIZE_t observation_id, SIZE_t last_node_id):
        self.proper_class = proper_class
        self.current_class = current_class
        self.observation_id = observation_id
        self.last_node_id = last_node_id

cdef class Tree:
    property children_left:
        def __get__(self):
            return self._get_node_ndarray()['left_child'][:self.node_count]

    property children_right:
        def __get__(self):
            return self._get_node_ndarray()['right_child'][:self.node_count]

    property parent:
        def __get__(self):
            return self._get_node_ndarray()['parent'][:self.node_count]

    property n_leaves:
        def __get__(self):
            return np.sum(np.logical_and(
                self.children_left == -1,
                self.children_right == -1))

    property feature:
        def __get__(self):
            return self._get_node_ndarray()['feature'][:self.node_count]

    property threshold:
        def __get__(self):
            return self._get_node_ndarray()['threshold'][:self.node_count]

    property nodes_depth:
        def __get__(self):
            return self._get_node_ndarray()['depth'][:self.node_count]

    def __cinit__(self, int n_features, int n_classes, DTYPE_t[:, :] thresholds, int initial_depth):
        """Constructor."""
        # Input/Output layout
        self.n_features = n_features
        self.n_classes = n_classes

        self.n_thresholds = thresholds.shape[0]
        self.thresholds = thresholds

        # Inner structures
        self.depth = 0
        self.node_count = 0
        self.capacity = 0
        self.nodes = NULL
        self.observations = {}   # dictionary from node id to list of observation struct

        self._resize_by_initial_depth(initial_depth)

    def __dealloc__(self):
        """Destructor."""
        # Free all inner structures
        free(self.nodes)

    def __reduce__(self):
        """Reduce re-implementation, for pickling."""
        return (Tree, (self.n_features,
               self.n_classes,
               np.array(self.thresholds),
               1), self.__getstate__())

    def __getstate__(self):
        """Getstate re-implementation, for pickling."""
        d = {}
        # capacity is inferred during the __setstate__ using nodes
        d["depth"] = self.depth
        d["node_count"] = self.node_count
        d["nodes"] = self._get_node_ndarray()
        # TODO uncomment after pull request with master
        # d["proper_classified"] = self.proper_classified
        d["observations"] = self.observations

    def __setstate__(self, d):
        """Setstate re-implementation, for unpickling."""
        self.depth = d["depth"]
        self.node_count = d["node_count"]
        # TODO uncomment after pull request with master
        # self.proper_classified = d["proper_classified"]
        self.observations = d["observations"]

        if 'nodes' not in d:
            raise ValueError('You have loaded Tree version which '
                             'cannot be imported')

        node_ndarray = d['nodes']

        if (node_ndarray.ndim != 1 or
                node_ndarray.dtype != NODE_DTYPE or
                not node_ndarray.flags.c_contiguous):
            raise ValueError('Did not recognise loaded array layout')

        self.capacity = node_ndarray.shape[0]
        if self._resize_c(self.capacity) != 0:
            raise MemoryError("resizing tree to %d" % self.capacity)
        nodes = memcpy(self.nodes, (<np.ndarray> node_ndarray).data,
                       self.capacity * sizeof(Node))

    cdef _resize_by_initial_depth(self, int initial_depth):
        if initial_depth <= 10:
            init_capacity = (2 ** (initial_depth + 1)) - 1
        else:
            init_capacity = 2047

        self._resize(init_capacity)

    cdef int _resize(self, SIZE_t capacity) nogil except -1:
        """Resize all inner arrays to `capacity`, if `capacity` == -1, then
           double the size of the inner arrays.
        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        if self._resize_c(capacity) != 0:
            # Acquire gil only if we need to raise
            with gil:
                raise MemoryError()

    cdef int _resize_c(self, SIZE_t capacity=SIZE_MAX) nogil except -1:
        """Guts of _resize
        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        if capacity == self.capacity and self.nodes != NULL:
            return 0

        if capacity == SIZE_MAX:
            if self.capacity == 0:
                capacity = 3  # default initial value
            else:
                capacity = 2 * self.capacity

        safe_realloc(&self.nodes, capacity)

        # if capacity smaller than node_count, adjust the counter
        if capacity < self.node_count:
            self.node_count = capacity

        self.capacity = capacity
        return 0

    cdef SIZE_t _add_node(self, SIZE_t parent, bint is_left, bint is_leaf,
                          SIZE_t feature, double threshold, SIZE_t depth,
                          SIZE_t class_number) nogil except -1:
        """Add a node to the tree.
        The new node registers itself as the child of its parent.
        Returns (size_t)(-1) on error.
        """
        cdef SIZE_t node_id = self.node_count

        if node_id >= self.capacity:
            if self._resize_c() != 0:
                return SIZE_MAX

        cdef Node* node = &self.nodes[node_id]

        node.parent = parent
        node.depth = depth

        if parent != _TREE_UNDEFINED:
            if is_left:
                self.nodes[parent].left_child = node_id
            else:
                self.nodes[parent].right_child = node_id

        if is_leaf:
            node.left_child = _TREE_LEAF
            node.right_child = _TREE_LEAF
            node.feature = class_number             # class instead of feature
            node.threshold = _TREE_UNDEFINED

        else:
            # left_child and right_child will be set later
            node.feature = feature
            node.threshold = threshold

        self.node_count += 1

        return node_id

    cdef np.ndarray _get_node_ndarray(self):
        """Wraps nodes as a NumPy struct array.
        The array keeps a reference to this Tree, which manages the underlying
        memory. Individual fields are publicly accessible as properties of the
        Tree.
        """
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> self.node_count
        cdef np.npy_intp strides[1]
        strides[0] = sizeof(Node)
        cdef np.ndarray arr
        Py_INCREF(NODE_DTYPE)
        arr = PyArray_NewFromDescr(<PyTypeObject *> np.ndarray,
                                   <np.dtype> NODE_DTYPE, 1, shape,
                                   strides, <void*> self.nodes,
                                   np.NPY_DEFAULT, None)
        Py_INCREF(self)
        arr.base = <PyObject*> self
        return arr

# ===========================================================================================================
# Mutation functions
# ===========================================================================================================

    """
    Function to mutate random node in tree (it can be leaf or decision node)
    In case the selected node is leaf it changes class
    In other case it changes feature and threshold
    """
    cpdef mutate_random_node(self):
        if self.node_count == 0:  # empty tree
            return
        cdef SIZE_t node_id = self.get_random_node()
        if self.nodes[node_id].left_child == _TREE_LEAF:
            self._mutate_class(node_id)
        else:
            self._mutate_feature(node_id)

    """
    Function to mutate random node in tree (it can be leaf or decision node)
    In case the selected node is leaf it changes class
    In other case it changes only threshold
    """
    cpdef mutate_random_class_or_threshold(self):
        if self.node_count == 0:  # empty tree
            return
        cdef SIZE_t node_id = self.get_random_node()
        if self.nodes[node_id].left_child == _TREE_LEAF:
            self._mutate_class(node_id)
        else:
            self._mutate_threshold(node_id, 0)

    """
    Function to mutate random decision node by changing feature and threshold
    """
    cpdef mutate_random_feature(self):
        if self.node_count <= 1:  # there is only one or 0 leaf
            return
        self._mutate_feature(self._get_random_decision_node())

    """
    Function to mutate random decision node by changing only threshold
    """
    cpdef mutate_random_threshold(self):
        if self.node_count <= 1:  # there is only one or 0 leaf
            return
        self._mutate_threshold(self._get_random_decision_node(), 0)

    """
    Function to mutate random leaf by changing class
    """
    cpdef mutate_random_class(self):
        if self.node_count == 0:  # empty tree
            return
        self._mutate_class(self._get_random_leaf())

    cdef _mutate_feature(self, SIZE_t node_id):
        cdef SIZE_t feature = self._get_new_random_feature(self.nodes[node_id].feature)
        self._change_feature_or_class(node_id, feature)
        self._mutate_threshold(node_id, 1)

    cdef _mutate_threshold(self, SIZE_t node_id, bint feature_changed):
        cdef DOUBLE_t threshold = self._get_new_random_threshold(self.nodes[node_id].threshold, self.nodes[node_id].feature, feature_changed)
        self._change_threshold(node_id, threshold)
        self._remove_observations_below_node(node_id)

    cdef _mutate_class(self, SIZE_t node_id):
        cdef SIZE_t new_class = self._get_new_random_class(self.nodes[node_id].feature)
        self._change_feature_or_class(node_id, new_class)
        self._remove_observations_below_node(node_id)

    cdef public SIZE_t get_random_node(self):
        cdef SIZE_t random_id = np.random.randint(0, self.node_count)
        return random_id

    cdef SIZE_t _get_random_decision_node(self):
        cdef SIZE_t random_id = np.random.randint(0, self.node_count)
        while self.nodes[random_id].left_child == _TREE_LEAF:
            random_id = np.random.randint(0, self.node_count)
        return random_id

    cdef SIZE_t _get_random_leaf(self):
        cdef SIZE_t random_id = np.random.randint(0, self.node_count)
        while self.nodes[random_id].left_child != _TREE_LEAF:
            random_id = np.random.randint(0, self.node_count)
        return random_id

    cdef SIZE_t _get_new_random_feature(self, SIZE_t last_feature):
        cdef SIZE_t new_feature = np.random.randint(0, self.n_features-1)
        if new_feature >= last_feature:
            new_feature += 1
        return new_feature

    cdef DOUBLE_t _get_new_random_threshold(self, DOUBLE_t last_threshold, SIZE_t feature, bint feature_changed):
        cdef SIZE_t new_threshold_index
        if feature_changed == 1:
            new_threshold_index = np.random.randint(0, self.n_thresholds)
        else:
            new_threshold_index = np.random.randint(0, self.n_thresholds-1)
            if self.thresholds[new_threshold_index, feature] >= last_threshold:
                new_threshold_index += 1
        return self.thresholds[new_threshold_index, feature]

    cdef SIZE_t _get_new_random_class(self, SIZE_t last_class):
        cdef SIZE_t new_class = np.random.randint(0, self.n_classes-1)
        if new_class >= last_class:
            new_class += 1
        return new_class

    cdef _change_feature_or_class(self, SIZE_t node_id, SIZE_t new_feature):
        self.nodes[node_id].feature = new_feature

    cdef _change_threshold(self, SIZE_t node_id, DOUBLE_t new_threshold):
        self.nodes[node_id].threshold = new_threshold

# ===========================================================================================================
# Observations functions
# ===========================================================================================================
    # initialization of observations
    cpdef initialize_observations(self, object X, np.ndarray y):
        cdef SIZE_t node_id
        cdef SIZE_t proper_class
        cdef SIZE_t current_class
        cdef SIZE_t observation_id
        cdef Observation observation

        cdef DTYPE_t[:, :] X_ndarray = X

        for observation_id in range(y.shape[0]):
            node_id = self._find_leaf_for_observation(observation_id, X_ndarray, 0)
            proper_class = y[observation_id]
            current_class = self.nodes[node_id].feature
            observation = Observation(proper_class, current_class, observation_id, node_id)
            self._assign_leaf_for_observation(observation, node_id)

    # assigning only not registered observations (because of crossing or mutation)
    cpdef assign_all_not_registered_observations(self, object X):
        if not self.observations.__contains__(NOT_REGISTERED):
            return
        cdef SIZE_t node_id
        cdef list observations = self.observations[NOT_REGISTERED]

        cdef DTYPE_t[:, :] X_ndarray = X

        for observation in observations:
            node_id = self._find_leaf_for_observation(observation.observation_id,
                                                      X_ndarray, observation.last_node_id)
            observation.current_class = self.nodes[node_id].feature
            observation.last_node_id = node_id
            self._assign_leaf_for_observation(observation, node_id)
        self.observations[NOT_REGISTERED] = []

    # adding observation to proper leaf
    cdef _assign_leaf_for_observation(self, Observation observation, SIZE_t node_id):
        if self.observations.__contains__(node_id):
            self.observations[node_id].append(observation)
        else:
            self.observations[node_id] = [observation]

    # finding proper leaf for observation
    cdef SIZE_t _find_leaf_for_observation(self, SIZE_t observation_id, DTYPE_t[:, :] X_ndarray,
                                               SIZE_t node_id_to_start) nogil:
        cdef DTYPE_t[:] X_row = X_ndarray[observation_id, :]
        cdef SIZE_t current_node_id = node_id_to_start
        cdef SIZE_t feature
        cdef DOUBLE_t threshold
        with nogil:
            while self.nodes[current_node_id].left_child != _TREE_LEAF:
                feature = self.nodes[current_node_id].feature
                threshold = self.nodes[current_node_id].threshold
                if X_row[feature] <= threshold:
                    current_node_id = self.nodes[current_node_id].left_child
                else:
                    current_node_id = self.nodes[current_node_id].right_child
        return current_node_id

    # remove all observations below node (fe. node changed in mutation)
    cdef _remove_observations_below_node(self, SIZE_t node_id):
        self._remove_observations_of_node_recurrent(node_id, node_id)

    # and the recurrent version of above
    cdef _remove_observations_of_node_recurrent(self, SIZE_t current_node_id, SIZE_t node_id_as_last):
        self._remove_observations_of_node(current_node_id, node_id_as_last)
        cdef Node node = self.nodes[current_node_id]
        if node.left_child != _TREE_LEAF:
            self._remove_observations_of_node_recurrent(node.left_child, node_id_as_last)
        if node.right_child != _TREE_LEAF:
            self._remove_observations_of_node_recurrent(node.right_child, node_id_as_last)

    # the main function to reassign all observations that should be removed to NOT_REGISTERED node id
    cdef _remove_observations_of_node(self, SIZE_t current_node_id, SIZE_t node_id_as_last):
        if not self.observations.__contains__(current_node_id):
            return
        cdef list observations = self.observations[current_node_id]
        if not self.observations.__contains__(NOT_REGISTERED):
            self.observations[NOT_REGISTERED] = []
        for observation in observations:
            observation.last_node_id = node_id_as_last
            self.observations[NOT_REGISTERED].append(observation)
        self.observations[current_node_id] = []

# ===========================================================================================================
# Prediction functions
# ===========================================================================================================

    cdef prepare_tree_to_prediction(self):
        # TODO  consider if should change anything (maybe observations=NULL; thresholds=NULL)
        pass

    cpdef np.ndarray predict(self, object X):
        cdef DTYPE_t[:, :] X_ndarray = X
        cdef n_observations = X_ndarray.shape[0]
        cdef np.ndarray y = np.empty(n_observations, dtype=DOUBLE)
        cdef int observation_id

        for observation_id in range(n_observations):
            node_id = self._find_leaf_for_observation(observation_id, X_ndarray, 0)
            y[observation_id] = self.nodes[node_id].feature  # feature means class for leaf

        return y

# ===========================================================================================================
# Multithreading test functions
# ===========================================================================================================

    cpdef test_function_with_args_core(self, char* name, long long size, int print_size):
        cdef long long x = 0
        for j in range(print_size):
            for i in range(size):
                x += 1
                for _ in range(1000000000):
                    for _ in range(1000000000):
                        x += 100000000
                        for _ in range(100000000):
                            x -= 1
            print(name, x)

    # reference how to use functions with arguments
    cpdef test_function_with_args(self, char* name, long long size, int print_size):
        multiprocessing.Process(target=self.test_function_with_args_core, args=(name, size, print_size)).start()

    cpdef time_test2(self, long long size):
        for i in range(10):
            self.time_test(size)

    cpdef time_test(self, long long size):
        cdef int x = 2
        for i in range(size):
            x *= 2
            x += 1
        printf("", x)

    cpdef time_test_nogil(self, long long size):
        self._time_test_nogil_(size)

    # function to test time because of many iterations
    cdef void _time_test_nogil_(self, long long size) nogil:
        cdef int x = 2
        with nogil:
            for i in range(size):
                x *= 2
                x += 1
            printf("", x)
