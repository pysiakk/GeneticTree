from enum import Enum, auto
from tree.builder import Builder, FullTreeBuilder, SplitTreeBuilder
from tree.tree import Tree
import numpy as np


class Initialization(Enum):
    Full = auto()
    Half = auto()
    Split = auto()


class Initializer:
    """
    Initializer is responsible for initializing population

    Mainly it creates trees with random features and thresholds inside decision
    nodes and with random classes inside leaves. The depth is set or chosen
    randomly.

    Args:
        n_trees: number of trees to create
        initial_depth: depth of all trees (if initialization type use it)
        initialization: how to initialize trees
    """

    def __init__(self,
                 n_trees: int = 400, initial_depth: int = 1,
                 initialization: Initialization = Initialization.Split,
                 split_prob: float = 0.7,
                 **kwargs):
        self.n_trees: int = n_trees
        self.initial_depth: int = initial_depth
        self.initialization: Initialization = initialization
        self.split_prob: float = split_prob
        self.builder: Builder = self.initialize_builder()

    def initialize_builder(self):
        """
        Returns:
            Builder: cython builder to initialize new trees
        """
        if self.initialization == Initialization.Full\
                or self.initialization == Initialization.Half:
            return FullTreeBuilder()
        elif self.initialization == Initialization.Split:
            return SplitTreeBuilder()

    def set_params(self, initial_depth: int = None,
                   initialization: Initialization = None,
                   **kwargs):
        """
        Function to set new parameters for Initializer

        Arguments are the same as in __init__
        """
        if initial_depth is not None:
            self.initial_depth = initial_depth
        if initialization is not None:
            self.initialization = initialization

    def initialize(self, X, y, sample_weight, threshold):
        """
        Function to initialize forest

        Args:
            forest: Container with all trees
        """
        if self.initialization == Initialization.Full:
            trees = self.initialize_full(X, y, sample_weight, threshold)
        elif self.initialization == Initialization.Half:
            trees = self.initialize_half(X, y, sample_weight, threshold)
        elif self.initialization == Initialization.Split:
            trees = self.initialize_split(X, y, sample_weight, threshold)
        return trees

    def initialize_full(self, X, y, sample_weight, thresholds):
        """
        Function to initialize forest

        Args:
            forest: Container with all trees
        """
        trees = []
        tree: Tree
        classes: np.ndarray = np.unique(y)

        for tree_index in range(self.n_trees):
            tree: Tree = Tree(classes, X, y, sample_weight, thresholds, np.random.randint(10**8))
            tree.resize_by_initial_depth(self.initial_depth)
            self.builder.build(tree, self.initial_depth)
            tree.initialize_observations()
            trees.append(tree)
        return trees

    def initialize_half(self, X, y, sample_weight, thresholds):
        """
        Function to initialize forest

        Args:
            forest: Container with all trees
        """
        trees = []
        tree: Tree
        classes: np.ndarray = np.unique(y)

        for tree_index in range(self.n_trees):
            if tree_index % 2 == 0:
                tree: Tree = Tree(classes, X, y, sample_weight, thresholds, np.random.randint(10**8))
                tree.resize_by_initial_depth(self.initial_depth)
                self.builder.build(tree, self.initial_depth)
                tree.initialize_observations()
                trees.append(tree)
            else:
                if self.initial_depth > 1:
                    depth = np.random.randint(low=1, high=self.initial_depth)
                else:
                    depth = self.initial_depth
                tree: Tree = Tree(classes, X, y, sample_weight, thresholds, np.random.randint(10**8))
                tree.resize_by_initial_depth(depth)
                self.builder.build(tree, depth)
                tree.initialize_observations()
                trees.append(tree)
        return trees

    def initialize_split(self, X, y, sample_weight, thresholds):
        """
        Function to initialize forest

        Args:
            forest: Container with all trees
        """
        trees = []
        tree: Tree
        classes: np.ndarray = np.unique(y)

        for tree_index in range(self.n_trees):
            tree: Tree = Tree(classes, X, y, sample_weight, thresholds, np.random.randint(10**8))
            tree.resize_by_initial_depth(self.initial_depth)
            self.builder.build(tree, self.initial_depth, self.split_prob)
            tree.initialize_observations()
            trees.append(tree)
        return trees
