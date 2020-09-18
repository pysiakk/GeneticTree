from tree.tree cimport Tree
import numpy as np
from tree._utils cimport Stack, StackRecord

from libc.stdlib cimport free
from libc.stdlib cimport malloc
from libc.stdio cimport printf

TREE_LEAF = -1
TREE_UNDEFINED = -2
cdef SIZE_t _TREE_LEAF = TREE_LEAF
cdef SIZE_t _TREE_UNDEFINED = TREE_UNDEFINED
cdef SIZE_t INITIAL_STACK_SIZE = 10

cdef class TreeCrosser:

    def __cinit__(self):
        pass

    cpdef Tree cross_trees(self, Tree first_parent, Tree second_parent):
        cdef SIZE_t first_node_id = first_parent.get_random_node()
        cdef SIZE_t second_node_id = second_parent.get_random_node()

        return self._cross_trees(first_parent, second_parent,
                                 first_node_id, second_node_id)

    # function to not random tests
    cpdef Tree _cross_trees(self, Tree first_parent, Tree second_parent,
                            SIZE_t first_node_id, SIZE_t second_node_id):
        cdef Tree child = self._initialize_new_tree(first_parent)

        cdef CrossoverPoint* result = <CrossoverPoint*> malloc(sizeof(StackRecord))

        self._add_tree_nodes(first_parent, first_node_id, child, 1, result)
        self._add_tree_nodes(second_parent, second_node_id, child, 0, result)

        free(result)
        return child

    """
    Important note: 
    - master is a parent tree
    - slave is a child tree
    Names changed because inside tree parent means node above and child node below
    
    Procedure:
    1. Add root node from parent to stack
    While stack not empty repeat:
        a) Remove node from Stack
        b) Register in child tree
        c) Add right_child of node in parent tree to stack if conditions
        d) Add left_child of node in parent tree to stack if conditions
        conditions == node exist in parent tree and id of this node is not equal node_id
    """
    cdef _add_tree_nodes(self, Tree master, SIZE_t crossover_point,
                         Tree slave, bint is_first, CrossoverPoint* result):
        cdef SIZE_t new_parent_id = _TREE_UNDEFINED
        cdef SIZE_t old_self_id = 0
        cdef bint is_left = 0
        cdef bint is_leaf = 0
        cdef SIZE_t feature = master.nodes[0].feature
        cdef double threshold = master.nodes[0].threshold
        cdef SIZE_t depth = 0
        cdef SIZE_t class_number = 0

        # to use with second master
        cdef SIZE_t depth_addition = 0
        if is_first == 0:
            new_parent_id = result[0].new_parent_id
            old_self_id = crossover_point
            is_left = result[0].is_left
            feature = master.nodes[crossover_point].feature
            threshold = master.nodes[crossover_point].threshold
            depth = master.nodes[crossover_point].depth
            depth_addition = result[0].depth_addition - master.nodes[crossover_point].depth

        cdef SIZE_t max_depth_seen = 0
        cdef SIZE_t success_code = 0

        cdef Stack stack = Stack(INITIAL_STACK_SIZE)
        cdef StackRecord stack_record

        with nogil:
            # push root node onto stack
            success_code = stack.push(new_parent_id, old_self_id, is_left,
                                      feature, threshold, depth)
            if success_code == -1:
                # got return code -1 - out-of-memory
                with gil:
                    raise MemoryError()

            while not stack.is_empty():
                stack.pop(&stack_record)

                new_parent_id = stack_record.new_parent_id
                old_self_id = stack_record.old_self_id
                is_left = stack_record.is_left
                feature = stack_record.feature
                threshold = stack_record.threshold

                # depth addition when second master
                depth = stack_record.depth + depth_addition

                is_leaf = 1
                if master.nodes[old_self_id].right_child != _TREE_LEAF:
                    is_leaf = 0

                if is_leaf == 1:
                    class_number = feature

                # when to stop adding nodes for first master
                if old_self_id == crossover_point and is_first == 1:
                    # remember: new_parent_id and is_left
                    # and not register node
                    result[0].new_parent_id = new_parent_id
                    result[0].is_left = is_left
                    result[0].depth_addition = depth
                    continue

                new_parent_id = slave._add_node(new_parent_id, is_left, is_leaf,
                                                feature, threshold, depth, class_number)

                self._register_node_in_stack(master, new_parent_id,
                                             master.nodes[old_self_id].left_child, 1,
                                             stack)

                self._register_node_in_stack(master, new_parent_id,
                                             master.nodes[old_self_id].right_child, 0,
                                             stack)

                if depth > max_depth_seen:
                    max_depth_seen = depth

                if success_code >= 0:
                    success_code = slave._resize_c(slave.node_count)

                if success_code >= 0:
                    slave.max_depth = max_depth_seen

    cdef void _register_node_in_stack(self, Tree master,
                                      SIZE_t new_parent_id, SIZE_t old_self_id,
                                      bint is_left, Stack stack) nogil:
        if old_self_id != _TREE_LEAF:
            stack.push(new_parent_id, old_self_id, is_left,
                       master.nodes[old_self_id].feature,
                       master.nodes[old_self_id].threshold,
                       master.nodes[old_self_id].depth)

    cdef Tree _initialize_new_tree(self, Tree previous_tree):
        cdef int n_features = previous_tree.n_features
        cdef int n_classes = previous_tree.n_classes

        cdef Tree tree = Tree(n_features, n_classes, previous_tree.thresholds, previous_tree.max_depth)
        return tree
