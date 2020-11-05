from enum import Enum, auto
from tree.forest import Forest
from tree.builder import Builder, FullTreeBuilder
from tree.tree import Tree
from numpy.random import randint


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

    def initialize(self, forest: Forest):
        """
        Function to initialize forest

        Args:
            forest: Container with all trees
        """
        if self.initialization_type == InitializationType.Random:
            self.initialize_random(forest)
        elif self.initialization_type == InitializationType.Half:
            self.initialize_half(forest)

    def initialize_random(self, forest: Forest):
        """
        Function to initialize forest

        Args:
            forest: Container with all trees
        """
        tree: Tree
        for tree_index in range(self.n_trees):
            tree = forest.create_new_tree(self.initial_depth)
            self.initialize_tree(tree, self.initial_depth)
            forest.add_new_tree_and_initialize_observations(tree)

    def initialize_half(self, forest: Forest):
        """
        Function to initialize forest

        Args:
            forest: Container with all trees
        """
        tree: Tree
        for tree_index in range(self.n_trees):
            if tree_index % 2 == 0:
                tree = forest.create_new_tree(self.initial_depth)
                self.initialize_tree(tree, self.initial_depth)
                forest.add_new_tree_and_initialize_observations(tree)
            else:
                depth = randint(1, self.initial_depth)
                tree = forest.create_new_tree(depth)
                self.initialize_tree(tree, depth)
                forest.add_new_tree_and_initialize_observations(tree)

    def initialize_tree(self, tree: Tree, initial_depth: int):
        """
        Args:
            tree: Tree to initialize nodes
            initial_depth: Depth to which tree will be initialized
        """
        self.builder.build(tree, self.initial_depth)
