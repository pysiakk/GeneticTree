# cython: linetrace=True

from tree.tree cimport Tree, Node, copy_tree
from tree._utils cimport Stack, StackRecord, IntArray

from libc.stdlib cimport free
from libc.stdlib cimport malloc

import numpy as np
cimport numpy as np

ctypedef np.npy_float64 DOUBLE_t         # Type of y
ctypedef np.npy_intp SIZE_t              # Type for indices and counters

TREE_LEAF = -1
TREE_UNDEFINED = -2
cdef SIZE_t _TREE_LEAF = TREE_LEAF
cdef SIZE_t _TREE_UNDEFINED = TREE_UNDEFINED
cdef SIZE_t INITIAL_STACK_SIZE = 10


"""
Structure to define handy values in order to copy nodes from second parent to child
"""
cdef struct CrossoverPoint:
    SIZE_t new_parent_id        # id of node that should be a parent of first copied node from second tree
    bint is_left                # if copied node should be registered as left
    SIZE_t depth_addition       # what is the depth of copied node in child tree


"""
Function to cross 2 trees
First it initializes new tree
Then it call function to add all nodes from parents
At the end it initializes observations dictionary
"""
cpdef Tree cross_trees(Tree first_parent, Tree second_parent,
                       int first_node_id, int second_node_id):

    cdef CrossoverPoint* result
    cdef Tree child

    if first_node_id == 0:
        result = <CrossoverPoint*> malloc(sizeof(StackRecord))
        result[0].new_parent_id = _TREE_UNDEFINED
        result[0].is_left = 0
        result[0].depth_addition = 0
        child = Tree(first_parent.n_classes, first_parent.X, first_parent.y, first_parent.thresholds)
        _copy_nodes(second_parent.nodes, second_node_id, child, result)
        child.initialize_observations()
        free(result)
        return child

    child = _initialize_new_tree(first_parent)

    _add_nodes_from_parents(child, first_parent.nodes, second_parent.nodes,
                            first_node_id, second_node_id)

    return child


"""
Function to cross 2 trees by defining a child tree
A child tree is created by replacing 
a subtree in first_parent from first_node_id 
by a subtree in second_parent from second_node_id
"""
cdef void _add_nodes_from_parents(Tree child,
                                  Node* first_parent_nodes, Node* second_parent_nodes,
                                  SIZE_t first_node_id, SIZE_t second_node_id):
    cdef CrossoverPoint* result = <CrossoverPoint*> malloc(sizeof(StackRecord))

    _remove_nodes_below_crossover_point(child, first_node_id, result)

    _copy_nodes(second_parent_nodes, second_node_id, child, result)

    result.new_parent_id = child._compress_removed_nodes(result.new_parent_id)

    cdef SIZE_t below_node_id = child.nodes[result.new_parent_id].right_child
    if result.is_left == 1:
        below_node_id = child.nodes[result.new_parent_id].left_child
    child.observations.reassign_observations(child, below_node_id)

    child._resize_c(child.node_count)

    free(result)


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
cdef _copy_nodes(Node* donor_nodes, SIZE_t crossover_point,
                 Tree recipient, CrossoverPoint* result):
    cdef SIZE_t new_parent_id = result[0].new_parent_id
    cdef SIZE_t old_self_id = crossover_point
    cdef bint is_left = result[0].is_left
    cdef bint is_leaf = 0
    cdef SIZE_t feature = donor_nodes[crossover_point].feature
    cdef double threshold = donor_nodes[crossover_point].threshold
    cdef SIZE_t depth = donor_nodes[crossover_point].depth
    cdef SIZE_t class_number = 0

    # to use with second donor
    cdef SIZE_t depth_addition = result[0].depth_addition - donor_nodes[crossover_point].depth

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
            if donor_nodes[old_self_id].left_child != _TREE_LEAF:
                is_leaf = 0

            if is_leaf == 1:
                class_number = feature

            new_parent_id = recipient._add_node(new_parent_id, is_left, is_leaf,
                                                feature, threshold, depth, class_number)

            _add_node_to_stack(donor_nodes, new_parent_id,
                               donor_nodes[old_self_id].left_child,
                               1, stack, donor_nodes[old_self_id].left_child)

            _add_node_to_stack(donor_nodes, new_parent_id,
                               donor_nodes[old_self_id].right_child,
                               0, stack, donor_nodes[old_self_id].left_child)

            if depth > max_depth_seen:
                max_depth_seen = depth

        if success_code >= 0:
            recipient.depth = max_depth_seen


cdef void _remove_nodes_below_crossover_point(Tree recipient, SIZE_t crossover_point,
                                              CrossoverPoint* result) nogil:
    cdef Node* nodes = recipient.nodes

    # remove observations
    with gil:
        recipient.observations.remove_observations(recipient.nodes, crossover_point)

    with nogil:
        result.new_parent_id = nodes[crossover_point].parent
        result.depth_addition = nodes[crossover_point].depth
        result.is_left = 0
        if nodes[result.new_parent_id].left_child == crossover_point:
            result.is_left = 1

    with gil:
        recipient._add_node_as_removed(crossover_point)


"""
Adds node with id old_self_id to Stack
"""
cdef void _add_node_to_stack(Node* donor_nodes,
                             SIZE_t new_parent_id, SIZE_t old_self_id,
                             bint is_left, Stack stack, SIZE_t check_leaf) nogil:
    if check_leaf != _TREE_LEAF:
        stack.push(new_parent_id, old_self_id, is_left,
                   donor_nodes[old_self_id].feature,
                   donor_nodes[old_self_id].threshold,
                   donor_nodes[old_self_id].depth)


"""
Creates new tree with base params as previous tree
"""
cdef Tree _initialize_new_tree(Tree previous_tree):
    return copy_tree(previous_tree)
