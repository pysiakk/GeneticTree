# (LICENSE) based on the same file as tree.pyx

from .tree cimport Tree

from libc.stdint cimport SIZE_MAX

import numpy as np
cimport numpy as np
cimport cython
np.import_array()

ctypedef np.npy_float32 DTYPE_t          # Type of X
ctypedef np.npy_float64 DOUBLE_t         # Type of y, sample_weight
ctypedef np.npy_intp SIZE_t              # Type for indices and counters
ctypedef np.npy_int32 INT32_t            # Signed 32 bit integer
ctypedef np.npy_uint32 UINT32_t          # Unsigned 32 bit integer

TREE_LEAF = -1
TREE_UNDEFINED = -2
cdef SIZE_t _TREE_LEAF = TREE_LEAF
cdef SIZE_t _TREE_UNDEFINED = TREE_UNDEFINED

cpdef full_tree_builder(Tree tree, int initial_depth):
    """
    Build a full decision tree
    
    Args:
        tree: Tree object that we created node in
        initial_depth: Maximum depth of initialized tree
    """
    cdef SIZE_t parent = TREE_UNDEFINED
    cdef bint is_left = 0
    cdef bint is_leaf = 0

    cdef int rc = 0
    cdef int current_depth
    cdef int node_number
    cdef int current_node_number

    with nogil:
        for current_depth in range(initial_depth+1):
            if current_depth == initial_depth:
                is_leaf = 1

            if rc == -1:
                # got return code -1 - out-of-memory
                with gil:
                    raise MemoryError()

            is_left = 0
            for node_number in range(2**current_depth):
                if is_left == 1:
                    is_left = 0
                else:
                    is_left = 1

                if current_depth == 0:
                    parent = _TREE_UNDEFINED
                else:
                    current_node_number = 2**current_depth + node_number
                    parent = <SIZE_t> ((current_node_number-2+is_left) / 2)

                if current_depth == initial_depth:
                    is_leaf = 1
                else:
                    is_leaf = 0

                node_id = add_node(tree, parent, is_left, is_leaf, current_depth)

                if node_id == SIZE_MAX:
                    rc = -1
                    break

    tree.depth = initial_depth



cpdef split_tree_builder(Tree tree, int initial_depth, double split_prob):
    """
    Build a decision tree using split method
    
    Args:
        tree: Tree object that we created node in
        initial_depth: Maximum depth of initialized tree
        split_prob: Probability of creating a decision node (not creating a leaf)
    """
    cdef max_node_number = 2**initial_depth
    cdef SIZE_t parent = TREE_UNDEFINED
    cdef bint is_left = 0
    cdef bint is_leaf = 0

    cdef int rc = 0
    cdef int current_depth
    cdef int node_number
    cdef int right
    cdef int left

    # trzymaj tabelę nodeów, które nie są liściami i losowo lub po kolei wybieraj i zastanawiaj się czy splitować
    with nogil:
        node_number = _node_creation(tree, initial_depth, _TREE_UNDEFINED, 0, 1, 1, split_prob)
        if node_number == -1:
            with gil:
                raise MemoryError()

    tree.depth = node_number

cdef int _node_creation(Tree tree, int initial_depth, int parent, int current_depth,
                        int is_left, int is_root, double split_prob) nogil:
    """
    A helping function for a recurrent node creation in the split initialization method
    
    Args:
        tree: Tree object that we created node in
        initial_depth: Maximum depth of initialized tree
        parent: Parent node of a node that is created in a current iteration
        current_depth: Depth of a current node
        is_left: Indicator if this node is a left child of the parent node
        is_root: Indicator if this node is a root node
        split_prob: Probability of creating a decision node (not creating a leaf)

    Returns:
        Maximum depth of a tree found in a subtree of a current node
    """
    if current_depth == 0:
        is_leaf = 0
    elif current_depth == initial_depth:
        is_leaf = 1
    else:
        if tree.randint_c(0, 1000000000)/1000000000 < split_prob:
            is_leaf = 0
        else:
            is_leaf = 1

    node_id = add_node(tree, parent, is_left, is_leaf, current_depth)

    # if is_leaf == 1:
    #     class_number = tree.classes[tree.randint_c(0, tree.n_classes)]
    # else:
    #     feature = tree.randint_c(0, tree.n_features)
    #     threshold = tree.thresholds[tree.randint_c(0, tree.n_thresholds), feature]
    #
    # node_id = tree.add_node(parent, is_left, is_leaf, feature,
    #                  threshold, current_depth, class_number)



    if node_id == SIZE_MAX:
        rc = -1
        return -1

    if is_leaf == 0:
        right = _node_creation(tree, initial_depth, node_id, current_depth+1, 1, 0, split_prob)
        if right == -1:
            with gil:
                raise MemoryError()
        left = _node_creation(tree, initial_depth, node_id, current_depth+1, 0, 0, split_prob)
        if left == -1:
            with gil:
                raise MemoryError()
        return max(right, left)
    else:
        return current_depth

    return 0

cdef SIZE_t add_node(Tree tree, SIZE_t parent, bint is_left, bint is_leaf, SIZE_t current_depth) nogil:
    """
    Function for adding node to the tree
    
    Args:
        tree: Tree object that we created node in
        parent: Parent node of a node that is created in a current iteration
        is_left: Indicator if this node is a left child of the parent node
        is_leaf: Indicator if this node is a leaf
        current_depth: Depth of a current node

    Returns:
        Index of the created node in an array of nodes
    """
    cdef SIZE_t class_number
    cdef SIZE_t feature
    cdef DTYPE_t threshold

    if is_leaf == 1:
        class_number = tree.classes[tree.randint_c(0, tree.n_classes)]
    else:
        feature = tree.randint_c(0, tree.n_features)
        threshold = tree.thresholds[tree.randint_c(0, tree.n_thresholds), feature]
    return tree._add_node(parent, is_left, is_leaf, feature, threshold, current_depth, class_number)

cpdef test_add_node(Tree tree, SIZE_t parent, bint is_left, SIZE_t feature, double threshold, SIZE_t depth):
    tree._add_node(parent, is_left, 0, feature, threshold, depth, 0)

cpdef test_add_leaf(Tree tree, SIZE_t parent, bint is_left, SIZE_t leaf_class, SIZE_t depth):
    tree._add_node(parent, is_left, 1, 0, 0.0, depth, leaf_class)
