from enum import Enum, auto
from tree.builder import Builder, FullTreeBuilder
from tree.tree import Tree
import numpy as np


class InitializationType(Enum):
    Random = auto()
    Half = auto()


class Initializer:
    """
    Initializer is responsible for initializing population

    Mainly it creates trees with random features and thresholds inside decision
    nodes and with random classes inside leaves. The depth is set or chosen
    randomly.

    Args:
        n_trees: number of trees to create
        initial_depth: depth of all trees (if initialization type use it)
        initialization_type: how to initialize trees
    """

    def __init__(self,
                 n_trees: int = 200, initial_depth: int = 1,
                 initialization_type: InitializationType = InitializationType.Random,
                 **kwargs):
        self.n_trees: int = n_trees
        self.initial_depth: int = initial_depth
        self.initialization_type: InitializationType = initialization_type
        self.builder: Builder = self.initialize_builder()

    def initialize_builder(self):
        """
        Returns:
            Builder: cython builder to initialize new trees
        """
        if self.initialization_type == InitializationType.Random:
            return FullTreeBuilder()

    def set_params(self, initial_depth: int = None,
                   initialization_type: InitializationType = None,
                   **kwargs):
        """
        Function to set new parameters for Initializer

        Arguments are the same as in __init__
        """
        if initial_depth is not None:
            self.initial_depth = initial_depth
        if initialization_type is not None:
            self.initialization_type = initialization_type

    def initialize(self, X, y, threshold):
        """
        Function to initialize forest

        Args:
            forest: Container with all trees
        """
        if self.initialization_type == InitializationType.Random:
            self.initialize_random(X, y, threshold)
        elif self.initialization_type == InitializationType.Half:
            self.initialize_half(X, y, threshold)

    def initialize_random(self, X, y, thresholds):
        """
        Function to initialize forest

        Args:
            forest: Container with all trees
        """
        trees = []
        tree: Tree
        n_classes: int = np.unique(y).shape[0]

        for tree_index in range(self.n_trees):
            tree: Tree = Tree(n_classes, X, y, thresholds)
            tree.resize_by_initial_depth(self.initial_depth)
            self.builder.build(tree, self.initial_depth)
            tree.initialize_observations()
            trees.append(tree)
        return trees

    def initialize_half(self, X, y, thresholds):
        """
        Function to initialize forest

        Args:
            forest: Container with all trees
        """
        trees = []
        tree: Tree
        n_classes: int = np.unique(y).shape[0]

        for tree_index in range(self.n_trees):
            if tree_index % 2 == 0:
                tree: Tree = Tree(n_classes, X, y, thresholds)
                tree.resize_by_initial_depth(self.initial_depth)
                self.builder.build(tree)
                tree.initialize_observations()
                trees.append(tree)
            else:
                depth = np.randint(1, self.initial_depth)
                tree: Tree = Tree(n_classes, X, y, thresholds)
                tree.resize_by_initial_depth(depth)
                self.builder.build(tree, depth)
                tree.initialize_observations()
                trees.append(tree)