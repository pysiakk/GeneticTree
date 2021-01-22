from aenum import Enum, extend_enum
from ..tree.builder import full_tree_builder, split_tree_builder
from ..tree.tree import Tree
import numpy as np
import warnings


def _initialize(X, y, sample_weight, thresholds, initializer, tree_builder, half):
    """
    A helping function for initializing trees. Its main purpose is to reduce the code repetition. Some
    initialization methods may use it with proper parameters to execute specific types of the initialization.

    Args:
        X: dataset to train model on as matrix of shape [n_observations x n_features]
        y: proper class of each observation as vector of shape [n_observations]
        sample_weight: a weight of each observation or None (meaning each observation have the same weight)
        thresholds: array of thresholds for particular dataset
        initializer: main initialization object containing crucial information about building trees
        tree_builder: function that builds trees during initialization (imported from tree.builder)
        half: indicator if trees should be initialized with a half method (half of the trees is initialized to
        the maximum depth and the other half is initialized to a random depth lower or equal than the maximum depth)

        Returns:
            An array of an initial population of trees
     """
    trees = []
    tree: Tree
    classes: np.ndarray = np.unique(y)
    kwargs = {}
    if tree_builder == split_tree_builder:
        kwargs = {"split_prob": initializer.split_prob}

    for tree_index in range(initializer.n_trees):
        if tree_index % 2 == 0 or not half:
            tree: Tree = Tree(classes, X, y, sample_weight, thresholds, np.random.randint(10 ** 8))
            tree.resize_by_initial_depth(initializer.initial_depth)
            tree_builder(tree, initializer.initial_depth, **kwargs)
            tree.initialize_observations()
            trees.append(tree)
        else:
            if initializer.initial_depth > 1:
                depth = np.random.randint(low=1, high=initializer.initial_depth)
            else:
                depth = initializer.initial_depth
            tree: Tree = Tree(classes, X, y, sample_weight, thresholds, np.random.randint(10 ** 8))
            tree.resize_by_initial_depth(depth)
            tree_builder(tree, depth, **kwargs)
            tree.initialize_observations()
            trees.append(tree)
    return trees


def initialize_full(X, y, sample_weight, thresholds, initializer):
    """
    Function that executes the initialization with a full method.

    Args:
        X: dataset to train model on as matrix of shape [n_observations x n_features]
        y: proper class of each observation as vector of shape [n_observations]
        sample_weight: a weight of each observation or None (meaning each observation have the same weight)
        thresholds: array of thresholds for particular dataset
        initializer: main initialization object containing crucial information about building trees

        Returns:
            An array of an initial population of trees
    """
    return _initialize(X, y, sample_weight, thresholds, initializer, full_tree_builder, False)


def initialize_half(X, y, sample_weight, thresholds, initializer):
    """
    Function that executes the initialization with a half method.

    Args:
        X: dataset to train model on as matrix of shape [n_observations x n_features]
        y: proper class of each observation as vector of shape [n_observations]
        sample_weight: a weight of each observation or None (meaning each observation have the same weight)
        thresholds: array of thresholds for particular dataset
        initializer: main initialization object containing crucial information about building trees

        Returns:
            An array of an initial population of trees
    """
    return _initialize(X, y, sample_weight, thresholds, initializer, full_tree_builder, True)


def initialize_split(X, y, sample_weight, thresholds, initializer):
    """
    Function that executes the initialization with a split method.

    Args:
        X: dataset to train model on as matrix of shape [n_observations x n_features]
        y: proper class of each observation as vector of shape [n_observations]
        sample_weight: a weight of each observation or None (meaning each observation have the same weight)
        thresholds: array of thresholds for particular dataset
        initializer: main initialization object containing crucial information about building trees

        Returns:
            An array of an initial population of trees
    """
    return _initialize(X, y, sample_weight, thresholds, initializer, split_tree_builder, False)


class Initialization(Enum):
    """
    Initialization is enumerator with possible initialization methods to use:
        Full -- every tree is build fully to exactly the same depth
        Half -- every tree is build full, but half of the trees has the maximum depth and the other half has a random
        depth lower or equal than the maximum depth
        Split -- every tree is build by randomly deciding in each node (beginning with the root decision node) if this
        node should be a decision node or a leaf (split_prob is a probability of creating a decision node)

    To add new Initialization method see genetic.initialization.Initialization
    """

    def __new__(cls, function, *args):
        obj = object.__new__(cls)
        obj._value_ = len(cls.__members__)
        obj.initialize = function
        return obj

    @staticmethod
    def add_new(name, function):
        extend_enum(Initialization, name, function)

    # after each entry should be at least delimiter
    # (also can be more arguments which will be ignored)
    # this is needed because value is callable type
    Full = initialize_full,
    Half = initialize_half,
    Split = initialize_split,


class Initializer:
    """
    Initializer is responsible for initializing population

    Mainly it creates trees with random features and thresholds inside decision
    nodes and with random classes inside leaves. The depth is set or chosen
    randomly.

    Args:
        n_trees: number of trees to create
        initial_depth: maximum depth of the trees (if initialization type use it)
        initialization: how to initialize trees
        split_prob: probability of creating a decision node during initialization (only viable for the split
        initialization method)
    """

    def __init__(self,
                 n_trees: int = 400, initial_depth: int = 1,
                 initialization: Initialization = Initialization.Split,
                 split_prob: float = 0.7,
                 **kwargs):
        self.n_trees: int = self._check_n_trees(n_trees)
        self.initial_depth: int = self._check_initial_depth(initial_depth)
        self.initialization: Initialization = self._check_initialization(initialization)
        self.split_prob: float = self._check_split_prob(split_prob)

    @staticmethod
    def _check_initialization(initialization):
        # comparison of strings because after using Selection.add_new() Selection is reference to other class
        if str(type(initialization)) == str(Initialization):
            return initialization
        else:
            raise TypeError(f"Passed selection={initialization} with type {type(initialization)}, "
                            f"Needed argument with type Selection")

    @staticmethod
    def _check_n_trees(n_trees):
        if n_trees <= 0:
            warnings.warn(f"Try to set n_trees={n_trees}. Changed to n_trees=1, "
                          f"but try to set n_trees manually for value at least 20")
            n_trees = 1
        return n_trees

    @staticmethod
    def _check_initial_depth(initial_depth):
        if initial_depth <= 0:
            warnings.warn(f"Try to set initial_depth={initial_depth}. Changed to initial_depth=1.")
            initial_depth = 1
        return initial_depth

    @staticmethod
    def _check_split_prob(split_prob):
        if split_prob < 0 or split_prob > 1:
            warnings.warn(f"Try to set split_prob={split_prob}. Changed to initial_depth=0.7. "
                          f"Split prob should have value between 0 and 1")
            split_prob = 1
        return split_prob

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
            X: dataset to train model on as matrix of shape [n_observations x n_features]
            y: proper class of each observation as vector of shape [n_observations]
            sample_weight: a weight of each observation or None (meaning each observation have the same weight)
            thresholds: array of thresholds for particular dataset

        Returns:
            An array of an initial population of trees
        """
        return self.initialization.initialize(X, y, sample_weight, threshold, self)

