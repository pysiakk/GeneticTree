import numpy as np
cimport numpy as np

from tree.tree import Tree
from tree.tree cimport Tree

ctypedef np.npy_intp SIZE_t             # Type for indices and counters
ctypedef np.npy_float64 DOUBLE_t        # Type of thresholds

TREE_LEAF = -1
cdef SIZE_t _TREE_LEAF = TREE_LEAF

"""
Function to mutate random node in tree (it can be leaf or decision node)
In case the selected node is leaf it changes class
In other case it changes feature and threshold
"""
cpdef mutate_random_node(Tree tree):
    if tree.nodes.count == 0:  # empty tree
        return
    cdef SIZE_t node_id = tree.get_random_node()
    if tree.nodes.elements[node_id].left_child == _TREE_LEAF:
        _mutate_class(tree, node_id)
    else:
        _mutate_feature(tree, node_id)

"""
Function to mutate random node in tree (it can be leaf or decision node)
In case the selected node is leaf it changes class
In other case it changes only threshold
"""
cpdef mutate_random_class_or_threshold(Tree tree):
    if tree.nodes.count == 0:  # empty tree
        return
    cdef SIZE_t node_id = tree.get_random_node()
    if tree.nodes.elements[node_id].left_child == _TREE_LEAF:
        _mutate_class(tree, node_id)
    else:
        _mutate_threshold(tree, node_id, 0)

"""
Function to mutate random decision node by changing feature and threshold
"""
cpdef mutate_random_feature(Tree tree):
    if tree.nodes.count <= 1:  # there is only one or 0 leaf
        return
    _mutate_feature(tree, tree.get_random_decision_node())

"""
Function to mutate random decision node by changing only threshold
"""
cpdef mutate_random_threshold(Tree tree):
    if tree.nodes.count <= 1:  # there is only one or 0 leaf
        return
    _mutate_threshold(tree, tree.get_random_decision_node(), 0)

"""
Function to mutate random leaf by changing class
"""
cpdef mutate_random_class(Tree tree):
    if tree.nodes.count == 0:  # empty tree
        return
    _mutate_class(tree, tree.get_random_leaf())

cdef _mutate_feature(Tree tree, SIZE_t node_id):
    cdef SIZE_t feature = tree.get_new_random_feature(tree.nodes.elements[node_id].feature)
    tree.change_feature_or_class(node_id, feature)
    _mutate_threshold(tree, node_id, 1)

cdef _mutate_threshold(Tree tree, SIZE_t node_id, bint feature_changed=0):
    tree.observations.remove_observations(tree.nodes.elements, node_id)
    cdef DOUBLE_t threshold = tree.get_new_random_threshold(tree.nodes.elements[node_id].threshold, tree.nodes.elements[node_id].feature, feature_changed)
    tree.change_threshold(node_id, threshold)
    tree.observations.reassign_observations(tree, node_id)

cdef _mutate_class(Tree tree, SIZE_t node_id):
    tree.observations.remove_observations(tree.nodes.elements, node_id)
    cdef SIZE_t new_class = tree.get_new_random_class(tree.nodes.elements[node_id].feature)
    tree.change_feature_or_class(node_id, new_class)
    tree.observations.reassign_observations(tree, node_id)

