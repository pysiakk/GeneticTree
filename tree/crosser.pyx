# cython: linetrace=True

from tree.tree cimport Tree
from tree._utils cimport Stack, StackRecord

from libc.stdlib cimport free
from libc.stdlib cimport malloc

import numpy as np
cimport numpy as np

TREE_LEAF = -1
TREE_UNDEFINED = -2
cdef SIZE_t _TREE_LEAF = TREE_LEAF
cdef SIZE_t _TREE_UNDEFINED = TREE_UNDEFINED
cdef SIZE_t INITIAL_STACK_SIZE = 10

cdef class TreeCrosser:
    """
    TreeCrosser is responsible for crossing 2 trees with usage of Cython
    """

    def __cinit__(self):
        pass

    """
    Function to cross 2 trees
    
    If cross both then cross first tree with second and second with first 
    using the same crossing points
    """
    cpdef Tree[:] cross_trees(self, Tree first_parent, Tree second_parent,
                              bint cross_both):
        cdef SIZE_t first_node_id = first_parent.get_random_node()
        cdef SIZE_t second_node_id = second_parent.get_random_node()

        cdef Tree[:] trees = np.empty(2, Tree)

        trees[0] = self._cross_trees(first_parent, second_parent,
                                     first_node_id, second_node_id)

        if cross_both == 1:
            trees[1] = self._cross_trees(second_parent, first_parent,
                                         second_node_id, first_node_id)

        return trees

    """
    Function to cross 2 trees by defining a child tree
    A child tree is created by replacing 
    a subtree in first_parent from first_node_id 
    by a subtree in second_parent from second_node_id
    """
    cpdef Tree _cross_trees(self, Tree first_parent, Tree second_parent,
                            SIZE_t first_node_id, SIZE_t second_node_id):
        cdef Tree child = self._initialize_new_tree(first_parent)

        cdef CrossoverPoint* result = <CrossoverPoint*> malloc(sizeof(StackRecord))

        self._copy_nodes(first_parent, first_node_id, child, 1, result)
        self._copy_nodes(second_parent, second_node_id, child, 0, result)

        free(result)
        return child

    """
    Function copy nodes from parent to a child

    Important note: 
    - donor is a parent tree
    - recipient is a child tree
    Names changed because inside tree parent means node above and child node below
    
    Procedure:
    1. Add root node from parent to stack
    While stack not empty repeat:
        a) Remove node from Stack
        b) Register in child tree if this node is not crossover point (for first donor)
        c) Add right_child of node in parent tree to stack if conditions
        d) Add left_child of node in parent tree to stack if conditions
        conditions == node exist in parent tree
    """
    cdef _copy_nodes(self, Tree donor, SIZE_t crossover_point,
                     Tree recipient, bint is_first, CrossoverPoint* result):
        cdef SIZE_t new_parent_id = _TREE_UNDEFINED
        cdef SIZE_t old_self_id = 0
        cdef bint is_left = 0
        cdef bint is_leaf = 0
        cdef SIZE_t feature = donor.nodes[0].feature
        cdef double threshold = donor.nodes[0].threshold
        cdef SIZE_t depth = 0
        cdef SIZE_t class_number = 0

        # to use with second donor
        cdef SIZE_t depth_addition = 0
        if is_first == 0:
            new_parent_id = result[0].new_parent_id
            old_self_id = crossover_point
            is_left = result[0].is_left
            feature = donor.nodes[crossover_point].feature
            threshold = donor.nodes[crossover_point].threshold
            depth = donor.nodes[crossover_point].depth
            depth_addition = result[0].depth_addition - donor.nodes[crossover_point].depth

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

                # depth addition when second donor
                depth = stack_record.depth + depth_addition

                is_leaf = 1
                if donor.nodes[old_self_id].right_child != _TREE_LEAF:
                    is_leaf = 0

                if is_leaf == 1:
                    class_number = feature

                # when to stop adding nodes for first donor
                if old_self_id == crossover_point and is_first == 1:
                    # remember: new_parent_id and is_left
                    # and not register node
                    result[0].new_parent_id = new_parent_id
                    result[0].is_left = is_left
                    result[0].depth_addition = depth
                    continue

                new_parent_id = recipient._add_node(new_parent_id, is_left, is_leaf,
                                                feature, threshold, depth, class_number)

                self._add_node_to_stack(donor, new_parent_id,
                                        donor.nodes[old_self_id].left_child,
                                        1, stack)

                self._add_node_to_stack(donor, new_parent_id,
                                        donor.nodes[old_self_id].right_child,
                                        0, stack)

                if depth > max_depth_seen:
                    max_depth_seen = depth

                if success_code >= 0:
                    success_code = recipient._resize_c(recipient.node_count)

                if success_code >= 0:
                    recipient.depth = max_depth_seen

    """
    Adds node with id old_self_id to Stack
    """
    cdef void _add_node_to_stack(self, Tree donor,
                                 SIZE_t new_parent_id, SIZE_t old_self_id,
                                 bint is_left, Stack stack) nogil:
        if old_self_id != _TREE_LEAF:
            stack.push(new_parent_id, old_self_id, is_left,
                       donor.nodes[old_self_id].feature,
                       donor.nodes[old_self_id].threshold,
                       donor.nodes[old_self_id].depth)

    """
    Creates new tree with base params as previous tree
    """
    cdef Tree _initialize_new_tree(self, Tree previous_tree):
        cdef int n_features = previous_tree.n_features
        cdef int n_classes = previous_tree.n_classes
        cdef int depth = previous_tree.depth

        cdef Tree tree = Tree(n_features, n_classes, previous_tree.thresholds, depth)
        return tree
