from enum import Enum, auto
from tree.forest import Forest
from tree.builder import Builder, FullTreeBuilder
from tree.tree import Tree
from joblib import Parallel, delayed


class InitializationType(Enum):
    Random = auto()


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
            return FullTreeBuilder(self.initial_depth)

    def set_params(self, initial_depth: int = None,
                   initialization_type: InitializationType = None):
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
        Parallel(n_jobs=4, backend="threading", prefer="threads")(delayed(self._initialize)(forest)
                                                                  for _ in range(self.n_trees))

    def _initialize(self, forest):
        """
        Parallelization helper for initialize function

        Args:
            forest: Container with all trees
        """
        tree = forest.create_new_tree(self.initial_depth)
        self.initialize_tree(tree)
        forest.add_new_tree_and_initialize_observations(tree)

    def initialize_tree(self, tree: Tree):
        """
        Args:
            tree: Tree to initialize nodes
        """
        if self.initialization_type == InitializationType.Random:
            self.initialize_tree_by_random_initialization(tree)

    def initialize_tree_by_random_initialization(self, tree: Tree):
        """
        Args:
            tree: Tree to initialize nodes
        """
        self.builder.build(tree)
