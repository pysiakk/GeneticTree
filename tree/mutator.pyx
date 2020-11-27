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
    _mutate_feature(tree, _get_random_decision_node(tree))

"""
Function to mutate random decision node by changing only threshold
"""
cpdef mutate_random_threshold(Tree tree):
    if tree.nodes.count <= 1:  # there is only one or 0 leaf
        return
    _mutate_threshold(tree, _get_random_decision_node(tree), 0)

"""
Function to mutate random leaf by changing class
"""
cpdef mutate_random_class(Tree tree):
    if tree.nodes.count == 0:  # empty tree
        return
    _mutate_class(tree, _get_random_leaf(tree))

cdef _mutate_feature(Tree tree, SIZE_t node_id):
    cdef SIZE_t feature = _get_new_random_feature(tree, tree.nodes.elements[node_id].feature)
    _change_feature_or_class(tree, node_id, feature)
    _mutate_threshold(tree, node_id, 1)

cdef _mutate_threshold(Tree tree, SIZE_t node_id, bint feature_changed):
    tree.observations.remove_observations(tree.nodes.elements, node_id)
    cdef DOUBLE_t threshold = _get_new_random_threshold(tree, tree.nodes.elements[node_id].threshold, tree.nodes.elements[node_id].feature, feature_changed)
    _change_threshold(tree, node_id, threshold)
    tree.observations.reassign_observations(tree, node_id)

cdef _mutate_class(Tree tree, SIZE_t node_id):
    tree.observations.remove_observations(tree.nodes.elements, node_id)
    cdef SIZE_t new_class = _get_new_random_class(tree, tree.nodes.elements[node_id].feature)
    _change_feature_or_class(tree, node_id, new_class)
    tree.observations.reassign_observations(tree, node_id)

cdef SIZE_t _get_random_decision_node(Tree tree):
    cdef SIZE_t random_id = np.random.randint(0, tree.nodes.count)
    while tree.nodes.elements[random_id].left_child == _TREE_LEAF:
        random_id = np.random.randint(0, tree.nodes.count)
    return random_id

cdef SIZE_t _get_random_leaf(Tree tree):
    cdef SIZE_t random_id = np.random.randint(0, tree.nodes.count)
    while tree.nodes.elements[random_id].left_child != _TREE_LEAF:
        random_id = np.random.randint(0, tree.nodes.count)
    return random_id

cdef SIZE_t _get_new_random_feature(Tree tree, SIZE_t last_feature):
    cdef SIZE_t new_feature = np.random.randint(0, tree.n_features-1)
    if new_feature >= last_feature:
        new_feature += 1
    return new_feature

cdef DOUBLE_t _get_new_random_threshold(Tree tree, DOUBLE_t last_threshold, SIZE_t feature, bint feature_changed):
    cdef SIZE_t new_threshold_index
    if feature_changed == 1:
        new_threshold_index = np.random.randint(0, tree.n_thresholds)
    else:
        new_threshold_index = np.random.randint(0, tree.n_thresholds-1)
        if tree.thresholds[new_threshold_index, feature] >= last_threshold:
            new_threshold_index += 1
    return tree.thresholds[new_threshold_index, feature]

cdef SIZE_t _get_new_random_class(Tree tree, SIZE_t last_class):
    cdef SIZE_t new_class = np.random.randint(0, tree.n_classes-1)
    if new_class >= last_class:
        new_class += 1
    return new_class

cdef _change_feature_or_class(Tree tree, SIZE_t node_id, SIZE_t new_feature):
    tree.nodes.elements[node_id].feature = new_feature

cdef _change_threshold(Tree tree, SIZE_t node_id, DOUBLE_t new_threshold):
    tree.nodes.elements[node_id].threshold = new_threshold
