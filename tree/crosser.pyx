# cython: linetrace=True

from tree.tree cimport Tree, Node, copy_tree
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
cdef struct BranchParent:
    SIZE_t id               # id of node that should be a parent of first copied node from second tree
    bint is_child_left      # if first copied node should be registered as left
    SIZE_t depth_addition   # what is the depth of copied node in child tree

"""
Function to cross 2 trees depends on first parents' node_id
If it is 0 -> it only cut branch
Else it crosses two trees
"""
cpdef Tree cross_trees(Tree first_parent, Tree second_parent,
                       int first_node_id, int second_node_id):

    cdef child

    if first_node_id == 0:
        child = _cut_branch(second_parent, second_node_id)

    else:
        child = copy_tree(first_parent)
        _cross_trees(child, first_parent.nodes.elements, second_parent.nodes.elements,
                     first_node_id, second_node_id)

    return child

"""
Function to cross 2 trees by defining a child tree
A child tree is created by replacing 
a subtree in first_parent from first_node_id 
by a subtree in second_parent from second_node_id
"""
cdef _cross_trees(Tree child,
                  Node* first_parent_nodes, Node* second_parent_nodes,
                  SIZE_t first_node_id, SIZE_t second_node_id):
    cdef BranchParent* branch_parent = <BranchParent*> malloc(sizeof(StackRecord))

    _remove_parent_nodes(child, first_node_id, branch_parent)
    _copy_nodes(second_parent_nodes, second_node_id, child, branch_parent)

    branch_parent.id = child.compact_removed_nodes(branch_parent.id)
    _reassign_observations(child, branch_parent)
    child.depth = np.max(child.nodes_depth)

    free(branch_parent)

    return child

"""
Cut branch from parent tree induced by node_id
Initialize observations in new created tree
Return new tree
"""
cdef Tree _cut_branch(Tree parent, int node_id):
    cdef BranchParent* result = <BranchParent*> malloc(sizeof(StackRecord))
    result[0].id = _TREE_UNDEFINED
    result[0].is_child_left = 0
    result[0].depth_addition = 0

    cdef Tree child = Tree(parent.n_classes, parent.X, parent.y, parent.sample_weight, parent.thresholds, np.random.randint(10**8))
    child.depth = 0
    _copy_nodes(parent.nodes.elements, node_id, child, result)
    child.initialize_observations()

    free(result)
    return child

"""
Function to remove nodes and observations from recipient below crossover point
"""
cdef void _remove_parent_nodes(Tree recipient, SIZE_t crossover_point,
                               BranchParent* result) nogil:
    cdef Node* nodes = recipient.nodes.elements

    with nogil:
        result.id = nodes[crossover_point].parent
        result.depth_addition = nodes[crossover_point].depth
        result.is_child_left = 0
        if nodes[result.id].left_child == crossover_point:
            result.is_child_left = 1

    # remove nodes and observations
    with gil:
        recipient.observations.remove_observations(recipient.nodes.elements, crossover_point)
        recipient.mark_nodes_as_removed(crossover_point)

"""
Function to assign observations in tree that was previously removed
"""
cdef void _reassign_observations(Tree child, BranchParent* branch_parent):
    cdef SIZE_t below_node_id = child.nodes.elements[branch_parent.id].right_child
    if branch_parent.is_child_left == 1:
        below_node_id = child.nodes.elements[branch_parent.id].left_child
    child.observations.reassign_observations(child, below_node_id)

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
                 Tree recipient, BranchParent* result):
    cdef SIZE_t new_parent_id = result[0].id
    cdef SIZE_t old_self_id = crossover_point
    cdef bint is_left = result[0].is_child_left
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
