from enum import Enum, auto
from tree.forest import Forest


class InitializationType(Enum):
    Random = auto()


class Initializer:
    """
    Class responsible for initializing population
    """

    def __init__(self, n_trees: int = 1000, max_depth: int = 3,
                 initialization_type: InitializationType = InitializationType.Random, **kwargs):
        self.n_trees: int = n_trees
        self.max_depth: int = max_depth
        self.initialization_type: InitializationType = initialization_type

    def set_params(self, n_trees: int = None, max_depth: int = None,
                   initialization_type: InitializationType = None):
        if n_trees is not None:
            self.n_trees = n_trees
        if max_depth is not None:
            self.max_depth = max_depth
        if initialization_type is not None:
            self.initialization_type = initialization_type

    def initialize(self, forest: Forest):
        if self.initialization_type == InitializationType.Random:
            forest.initialize_population(self.max_depth)
