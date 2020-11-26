# cython: linetrace=True

from tree.tree cimport Tree, Node
from tree._utils cimport Stack, StackRecord

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

    cdef Tree child = _initialize_new_tree(first_parent)

    _add_nodes_from_parents(child, first_parent.nodes, second_parent.nodes,
                            first_node_id, second_node_id)


    # TODO change below line to more complicated way that should be less time consuming
    # During copying nodes from first tree copy also all observations dict
    # and replace observations below changed node as NOT_REGISTERED
    # Then after completion of all tree only need to run assign_all_not_registered_observations
    child.initialize_observations()

    return child


"""
Function to run _cross_trees with testing purpose
"""
cpdef Tree test_cross_trees(Tree first_parent, Tree second_parent,
                            int first_node_id, int second_node_id):
    cdef Tree child = _initialize_new_tree(first_parent)
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

    _copy_nodes(first_parent_nodes, first_node_id, child, 1, result)
    _copy_nodes(second_parent_nodes, second_node_id, child, 0, result)

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
                 Tree recipient, bint is_first, CrossoverPoint* result):
    cdef SIZE_t new_parent_id = _TREE_UNDEFINED
    cdef SIZE_t old_self_id = 0
    cdef bint is_left = 0
    cdef bint is_leaf = 0
    cdef SIZE_t feature = donor_nodes[0].feature
    cdef double threshold = donor_nodes[0].threshold
    cdef SIZE_t depth = 0
    cdef SIZE_t class_number = 0

    # to use with second donor
    cdef SIZE_t depth_addition = 0
    if is_first == 0:
        new_parent_id = result[0].new_parent_id
        old_self_id = crossover_point
        is_left = result[0].is_left
        feature = donor_nodes[crossover_point].feature
        threshold = donor_nodes[crossover_point].threshold
        depth = donor_nodes[crossover_point].depth
        depth_addition = result[0].depth_addition - donor_nodes[crossover_point].depth

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
    cdef int n_classes = previous_tree.n_classes
    cdef int node_count = previous_tree.node_count

    cdef Tree tree = Tree(n_classes, previous_tree.X, previous_tree.y, previous_tree.thresholds)
    tree._resize(node_count)
    return tree
