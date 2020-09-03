from tree.tree import Tree

cdef class TreeCrosser:
    cdef Tree cross_trees(self, Tree first_parent, Tree second_parent)
