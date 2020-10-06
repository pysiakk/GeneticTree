from enum import Enum, auto
from tree.forest import Forest


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

    def __init__(self, initial_depth: int = 3,
                 initialization_type: InitializationType = InitializationType.Random, **kwargs):
        self.initial_depth: int = initial_depth
        self.initialization_type: InitializationType = initialization_type

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
        if self.initialization_type == InitializationType.Random:
            forest.initialize_population(self.initial_depth)
