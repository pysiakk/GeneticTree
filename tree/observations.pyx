from libc.stdlib cimport free
from libc.string cimport memcpy
from libc.stdint cimport SIZE_MAX
from libc.stdio cimport printf

from tree._utils cimport resize_c, resize

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
                  SIZE_t[:] y):
        self.n_observations = X.shape[0]
        self.proper_classified = 0

        self.X = X
        cdef DTYPE_t[:, :] X_ndarray = X
        self.X_ndarray = X_ndarray
        self.y = y

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

    def __reduce__(self):
        """Reduce re-implementation, for pickling."""
        # TODO
        pass

    def __getstate__(self):
        """Getstate re-implementation, for pickling."""
        # TODO
        pass

    def __setstate__(self, state):
        """Setstate re-implementation, for unpickling."""
        # TODO
        pass

    cpdef initialize_observations(self):
        # TODO
        # for each observation in y _assign_observation
        pass

    cpdef remove_observations(self, SIZE_t leaves_id):
        # TODO
        # for each observation minus one from proper_classified if proper classified
        # remember leaves_id (probably in new structure) that was removed
        # maybe in new structure have a pointer to IntArray
        # and then it can be possible to overwrite this index in Leaves
        # to properly overwrite can be needed second structure with indices to overwrite
        # and then all not overwritten indices should be moved to beginning (maybe - if possible)
        pass

    cpdef reassign_observations(self, SIZE_t below_node_id):
        # TODO
        # for each leaves_id in removed leaves_id _reassign_observations_for_leaf
        # maybe clean up some of above structures (i.e. old IntArray)
        pass

    cdef _reassign_observations_for_leaf(self, SIZE_t leaves_id, SIZE_t below_node_id):
        # TODO
        # for each observation find new leaf and _assign_observation
        pass

    cdef _assign_observation(self, Node* nodes, SIZE_t y_id, SIZE_t below_node_id):
        cdef node_id = self._find_leaf_for_observation(nodes, y_id, below_node_id)

        if nodes[node_id].right_child != TREE_LEAF:         # in right child is leaves_id
            self._append_observations(nodes[node_id].right_child, y_id)
        else:
            nodes[node_id].right_child = self._append_leaves(y_id)

        if nodes[node_id].feature == self.y[y_id]:          # feature means class
            self.proper_classified += 1

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

    cdef _copy_element_from_leaves_to_leaves_to_reassign(self):
        # TODO
        # in leaves (for IntArray:) set count and capacity to 0 and elements to NULL
        # in leaves_to_reassign add more capacity if needed
        pass

    cdef _delete_leaves_to_reassign(self):
        # TODO
        # free inner structures
        # free elements
        # set elements to NULL, counter and capacity o 0
        pass

    cdef SIZE_t _append_leaves(self, SIZE_t y_id):
        cdef SIZE_t leaves_id = self.leaves.count

        if leaves_id >= self.leaves.capacity:
            if resize_c(&self.leaves) != 0:
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

    cdef _push_empty_leaves_ids(self, SIZE_t leaves_id):
        cdef SIZE_t empty_leaves_ids_id = self.empty_leaves_ids.count

        if empty_leaves_ids_id >= self.empty_leaves_ids.capacity:
            if resize_c(&self.empty_leaves_ids) != 0:
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
        if resize(&self.empty_leaves_ids, self.empty_leaves_ids.count) != 0:
            return SIZE_MAX

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
