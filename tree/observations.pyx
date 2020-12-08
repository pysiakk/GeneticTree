from libc.stdlib cimport free
from libc.string cimport memcpy
from libc.stdint cimport SIZE_MAX

from tree._utils cimport resize_c, resize, copy_leaves, copy_int_array, safe_realloc

import numpy as np
cimport numpy as np
np.import_array()

TREE_LEAF = -1
TREE_UNDEFINED = -2
NOT_REGISTERED = -1
NOT_CLASSIFIED = -1
cdef SIZE_t _TREE_LEAF = TREE_LEAF
cdef SIZE_t _TREE_UNDEFINED = TREE_UNDEFINED
cdef SIZE_t _NOT_REGISTERED = NOT_REGISTERED

cdef class Observations:
    def __cinit__(self,
                  object X,
                  SIZE_t[:] y,
                  DTYPE_t[:] weights):
        self.n_observations = X.shape[0]
        self.proper_classified = 0

        self.X = X
        cdef DTYPE_t[:, :] X_ndarray = X
        self.X_ndarray = X_ndarray
        self.y = y
        self.weights = weights

        self.leaves = NULL
        safe_realloc(&self.leaves, 1)
        self.leaves_to_reassign = NULL
        safe_realloc(&self.leaves_to_reassign, 1)
        self.empty_leaves_ids = NULL
        safe_realloc(&self.empty_leaves_ids, 1)

        self.leaves.elements = NULL
        self.leaves.count = 0
        self.leaves.capacity = 0

        self.leaves_to_reassign.elements = NULL
        self.leaves_to_reassign.count = 0
        self.leaves_to_reassign.capacity = 0

        self.empty_leaves_ids.elements = NULL
        self.empty_leaves_ids.count = 0
        self.empty_leaves_ids.capacity = 0

    def __dealloc__(self):
        if self.leaves.elements != NULL:
            for i in range(self.leaves.count):
                free(self.leaves.elements[i].elements)
        free(self.leaves.elements)
        if self.leaves_to_reassign.elements != NULL:
            for i in range(self.leaves_to_reassign.count):
                free(self.leaves_to_reassign.elements[i].elements)
        free(self.leaves_to_reassign.elements)
        free(self.empty_leaves_ids.elements)
        free(self.leaves)
        free(self.leaves_to_reassign)
        free(self.empty_leaves_ids)

    def __reduce__(self):
        """Reduce re-implementation, for pickling."""
        # never pickle observations during fit
        # after unpickling need to pass pointers to X and y arrays
        empty_2d_array = np.empty((1, 1), dtype=np.float32)
        empty_1d_array_int = np.empty(1, dtype=np.intp)
        empty_1d_array = np.empty(1, dtype=np.float32)
        return (Observations,
                (empty_2d_array, empty_1d_array_int, empty_1d_array),
                self.__getstate__())

    def __getstate__(self):
        """Getstate re-implementation, for pickling."""
        state = {}

        state["proper_classified"] = self.proper_classified
        state["n_observations"] = self.n_observations

        if self.leaves_to_reassign.count != 0:
            raise ValueError("Pickle observations with leaves_to_reassign "
                             "not empty is not supported")

        # TODO: add pickling of leaves and empty_leaves_ids

        return state

    def __setstate__(self, state):
        """Setstate re-implementation, for unpickling."""
        self.proper_classified = state["proper_classified"]
        self.n_observations = state["n_observations"]

        # if 'leaves' not in state or 'empty_leaves_ids' not in state:
        #     raise ValueError('You have loaded Tree version which '
        #                      'cannot be imported')

        # TODO: add unpickling of leaves and empty_leaves_ids

    cdef void initialize_observations(self, Tree tree):
        cdef SIZE_t y_id
        cdef SIZE_t start_from_node_id = 0
        for y_id in range(self.n_observations):
            self._assign_observation(tree.nodes.elements, y_id, start_from_node_id)

    cdef void remove_observations(self, Node* nodes, SIZE_t below_node_id):
        if nodes[below_node_id].left_child == _TREE_LEAF:
            if nodes[below_node_id].right_child != TREE_LEAF:  # means there are at least one observation inside node
                self._remove_observations_in_leaf(nodes[below_node_id].right_child, nodes[below_node_id].feature)
                nodes[below_node_id].right_child = TREE_LEAF

        else:
            if nodes[below_node_id].left_child != _TREE_LEAF:
                self.remove_observations(nodes, nodes[below_node_id].left_child)

            if nodes[below_node_id].right_child != _TREE_LEAF:
                self.remove_observations(nodes, nodes[below_node_id].right_child)

    cdef void _remove_observations_in_leaf(self, SIZE_t leaves_id, SIZE_t leaf_class):
        cdef SIZE_t i
        cdef IntArray observations = self.leaves.elements[leaves_id]
        for i in range(observations.count):
            if self.y[observations.elements[i]] == leaf_class:
                self.proper_classified -= self.weights[observations.elements[i]]

        self._copy_element_from_leaves_to_leaves_to_reassign(leaves_id)

    cdef void reassign_observations(self, Tree tree, SIZE_t below_node_id):
        cdef SIZE_t i
        cdef SIZE_t j
        cdef IntArray* observations
        for i in range(self.leaves_to_reassign.count):
            observations = &self.leaves_to_reassign.elements[i]
            for j in range(observations.count):
                self._assign_observation(tree.nodes.elements, observations.elements[j], below_node_id)

        self._delete_leaves_to_reassign()
        self._resize_empty_leaves_ids()

    cdef _assign_observation(self, Node* nodes, SIZE_t y_id, SIZE_t below_node_id):
        cdef node_id = self._find_leaf_for_observation(nodes, y_id, below_node_id)

        if nodes[node_id].right_child != TREE_LEAF:         # in right child is leaves_id
            self._append_observations(nodes[node_id].right_child, y_id)
        else:
            nodes[node_id].right_child = self._append_leaves(y_id)

        if nodes[node_id].feature == self.y[y_id]:          # feature means class
            self.proper_classified += self.weights[y_id]

    cdef SIZE_t _find_leaf_for_observation(self, Node* nodes, SIZE_t y_id, SIZE_t below_node_id) nogil:
        cdef SIZE_t current_node_id = below_node_id
        cdef SIZE_t feature
        cdef DOUBLE_t threshold
        with nogil:
            while nodes[current_node_id].left_child != _TREE_LEAF:
                feature = nodes[current_node_id].feature
                threshold = nodes[current_node_id].threshold
                if self.X_ndarray[y_id, feature] <= threshold:
                    current_node_id = nodes[current_node_id].left_child
                else:
                    current_node_id = nodes[current_node_id].right_child
        return current_node_id

    cdef SIZE_t _append_leaves(self, SIZE_t y_id):
        cdef SIZE_t leaves_id = self._pop_empty_leaves_ids()

        if leaves_id == -1:  # it means there was anything to pop
            leaves_id = self.leaves.count
        else:                # it means there was something to pop
            self.leaves.count -= 1  # minus 1 from counter, because at the end of function it will be added +1

        if leaves_id >= self.leaves.capacity:
            if resize_c(self.leaves) != 0:
                return SIZE_MAX

        cdef IntArray* observations = &self.leaves.elements[leaves_id]

        observations.elements = NULL
        observations.count = 0
        observations.capacity = 0

        self._append_observations(leaves_id, y_id)

        self.leaves.count += 1
        return leaves_id

    cdef _append_observations(self, SIZE_t leaves_id, SIZE_t y_id):
        cdef IntArray* observations = &self.leaves.elements[leaves_id]

        cdef SIZE_t observations_id = observations.count

        if observations_id >= observations.capacity:
            if resize_c(observations) != 0:
                return SIZE_MAX

        cdef SIZE_t* observation = &observations.elements[observations_id]
        observation[0] = y_id

        observations.count += 1

    cdef _copy_element_from_leaves_to_leaves_to_reassign(self, SIZE_t leaves_id):
        cdef SIZE_t leaves_to_reassign_id = self.leaves_to_reassign.count

        if leaves_to_reassign_id >= self.leaves_to_reassign.capacity:
            if resize_c(self.leaves_to_reassign) != 0:
                return SIZE_MAX

        cdef IntArray* observations = &self.leaves_to_reassign.elements[leaves_to_reassign_id]

        observations.elements = self.leaves.elements[leaves_id].elements
        observations.count = self.leaves.elements[leaves_id].count
        observations.capacity = self.leaves.elements[leaves_id].capacity

        self.leaves_to_reassign.count += 1

        self.leaves.elements[leaves_id].elements = NULL
        self.leaves.elements[leaves_id].count = 0
        self.leaves.elements[leaves_id].capacity = 0

        self._push_empty_leaves_ids(leaves_id)

    cdef _delete_leaves_to_reassign(self):
        cdef SIZE_t i
        if self.leaves_to_reassign.elements != NULL:
            for i in range(self.leaves_to_reassign.count):
                free(self.leaves_to_reassign.elements[i].elements)
        free(self.leaves_to_reassign.elements)
        self.leaves_to_reassign.elements = NULL
        self.leaves_to_reassign.count = 0
        self.leaves_to_reassign.capacity = 0

    cdef _push_empty_leaves_ids(self, SIZE_t leaves_id):
        cdef SIZE_t empty_leaves_ids_id = self.empty_leaves_ids.count

        if empty_leaves_ids_id >= self.empty_leaves_ids.capacity:
            if resize_c(self.empty_leaves_ids) != 0:
                return SIZE_MAX

        cdef SIZE_t* leaves_id_ptr = &self.empty_leaves_ids.elements[empty_leaves_ids_id]
        leaves_id_ptr[0] = leaves_id

        self.empty_leaves_ids.count += 1

    cdef SIZE_t _pop_empty_leaves_ids(self):
        if self.empty_leaves_ids.count == 0:
            return -1

        self.empty_leaves_ids.count -= 1

        cdef SIZE_t* leaves_id_ptr = &self.empty_leaves_ids.elements[self.empty_leaves_ids.count]
        return leaves_id_ptr[0]

    cdef _resize_empty_leaves_ids(self):
        cdef SIZE_t new_size = 3
        if self.empty_leaves_ids.count > 3:
            new_size = self.empty_leaves_ids.count
        if resize(self.empty_leaves_ids, new_size) != 0:
            return SIZE_MAX

    cpdef test_initialization(self, Tree tree):
        self.initialize_observations(tree)
        assert self.leaves.count > 0
        assert self.leaves.capacity > 0
        assert self.leaves.elements[0].count > 0
        assert self.leaves.elements[0].capacity > 0
        assert self.leaves.elements[0].elements[0] == 0

    cpdef test_removing_and_reassigning(self, Tree tree):
        self.initialize_observations(tree)
        cdef DTYPE_t proper_classified = self.proper_classified
        cdef SIZE_t leaves_count = self.leaves.count
        assert self.leaves_to_reassign.count == 0
        self.remove_observations(tree.nodes.elements, 0)
        assert self.leaves_to_reassign.count == self.leaves.count == self.empty_leaves_ids.count == leaves_count
        assert self.proper_classified == 0
        self.reassign_observations(tree, 0)
        assert proper_classified == self.proper_classified
        assert self.leaves.count == leaves_count
        assert self.leaves_to_reassign.count == 0
        assert self.empty_leaves_ids.count == 0
        assert self.empty_leaves_ids.capacity == 3

    cpdef test_create_leaves_array_simple(self):
        self._append_leaves(1)
        self._append_leaves(3)
        self._append_leaves(2)
        assert self.leaves.count == 3
        assert self.leaves.capacity >= 3
        cdef SIZE_t i
        for i in range(3):
            assert self.leaves.elements[i].count == 1
            assert self.leaves.elements[i].capacity >= 1
        assert self.leaves.elements[0].elements[0] == 1
        assert self.leaves.elements[1].elements[0] == 3
        assert self.leaves.elements[2].elements[0] == 2

    cpdef test_create_leaves_array_complex(self):
        self._append_leaves(1)
        self._append_leaves(3)
        self._append_leaves(2)
        self._append_observations(1, 4)
        self._append_observations(0, 5)
        self._append_observations(2, 6)
        cdef SIZE_t i
        for i in range(100):
            self._append_observations(2, i + 10)
        assert self.leaves.count == 3
        assert self.leaves.capacity >= 3
        for i in range(2):
            assert self.leaves.elements[i].count == 2
            assert self.leaves.elements[i].capacity >= 2
        assert self.leaves.elements[2].count == 102
        assert self.leaves.elements[2].capacity >= 102
        assert self.leaves.elements[0].elements[0] == 1
        assert self.leaves.elements[0].elements[1] == 5
        assert self.leaves.elements[1].elements[0] == 3
        assert self.leaves.elements[1].elements[1] == 4
        assert self.leaves.elements[2].elements[0] == 2
        assert self.leaves.elements[2].elements[1] == 6
        for i in range(100):
            assert self.leaves.elements[2].elements[i+2] == i+10

    cpdef test_create_leaves_array_many(self):
        cdef SIZE_t i
        for i in range(100):
            self._append_leaves(i)
        assert self.leaves.count == 100
        assert self.leaves.capacity >= 100
        for i in range(100):
            assert self.leaves.elements[i].count == 1
            assert self.leaves.elements[i].capacity >= 1
            assert self.leaves.elements[i].elements[0] == i

    cpdef test_empty_leaves_ids(self):
        cdef SIZE_t i
        cdef SIZE_t value
        for i in range(100):
            self._push_empty_leaves_ids(i)
        assert self.empty_leaves_ids.count == 100
        assert self.empty_leaves_ids.capacity >= 100
        for i in range(50):
            value = self._pop_empty_leaves_ids()
            assert i+value == 99
        assert self.empty_leaves_ids.count == 50
        assert self.empty_leaves_ids.capacity >= 100
        self._resize_empty_leaves_ids()
        assert self.empty_leaves_ids.capacity == 50
        assert self.empty_leaves_ids.elements[49] == 49

    cpdef test_copy_to_leaves_to_reassign(self):
        self._append_leaves(1)
        self._append_leaves(3)
        self._append_leaves(2)
        self._append_observations(1, 4)
        self._append_observations(0, 5)
        self._append_observations(2, 6)
        cdef SIZE_t i
        for i in range(100):
            self._append_observations(1, i + 10)
        self._copy_element_from_leaves_to_leaves_to_reassign(1)
        self._copy_element_from_leaves_to_leaves_to_reassign(0)
        assert self.leaves.elements[1].count == 0
        assert self.leaves.elements[1].capacity == 0
        assert self.leaves.elements[1].elements == NULL
        assert self.leaves.elements[0].count == 0
        assert self.leaves.elements[0].capacity == 0
        assert self.leaves.elements[0].elements == NULL
        assert self.leaves.elements[2].count == 2
        assert self.leaves.elements[2].capacity >= 2
        assert self.leaves.elements[2].elements != NULL
        assert self.leaves.elements[2].elements[0] == 2
        assert self.leaves.elements[2].elements[1] == 6
        assert self.leaves_to_reassign.elements[0].count == 102
        assert self.leaves_to_reassign.elements[0].capacity >= 102
        assert self.leaves_to_reassign.elements[1].count == 2
        assert self.leaves_to_reassign.elements[1].capacity >= 2
        assert self.leaves_to_reassign.elements[1].elements[0] == 1
        assert self.leaves_to_reassign.elements[1].elements[1] == 5
        assert self.leaves_to_reassign.elements[0].elements[0] == 3
        assert self.leaves_to_reassign.elements[0].elements[1] == 4
        for i in range(100):
            assert self.leaves_to_reassign.elements[0].elements[i+2] == i+10
        assert self.empty_leaves_ids.count == 2
        assert self.empty_leaves_ids.capacity >= 2
        assert self.empty_leaves_ids.elements[0] == 1
        assert self.empty_leaves_ids.elements[1] == 0

    cpdef test_delete_leaves_to_reassign(self):
        self._append_leaves(1)
        self._append_leaves(3)
        self._append_observations(1, 4)
        self._append_observations(0, 5)
        cdef SIZE_t i
        for i in range(100):
            self._append_observations(1, i + 10)
        self._copy_element_from_leaves_to_leaves_to_reassign(1)
        self._copy_element_from_leaves_to_leaves_to_reassign(0)
        self._delete_leaves_to_reassign()
        assert self.leaves_to_reassign.count == 0
        assert self.leaves_to_reassign.capacity == 0
        assert self.leaves_to_reassign.elements == NULL


cpdef Observations copy_observations(Observations observations):
    cdef Observations observations_copied = Observations(observations.X, observations.y, observations.weights)
    copy_leaves(observations.leaves, observations_copied.leaves)
    copy_int_array(observations.empty_leaves_ids, observations_copied.empty_leaves_ids)
    copy_leaves(observations.leaves_to_reassign, observations_copied.leaves_to_reassign)
    observations_copied.proper_classified = observations.proper_classified
    return observations_copied

