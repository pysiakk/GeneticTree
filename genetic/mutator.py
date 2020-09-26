from tree.forest import Forest
from tree.tree import Tree
import math
import numpy as np


class Mutator:
    """
    Mutator mutate individuals.

    It provides interface to allow each type of mutation
    and to set different probability for each mutation.

    The probability means the proportion of all trees that be affected by mutation

    Args:
        is_feature: is feature mutation used
        feature_prob: probability of feature mutation
        is_threshold: is threshold mutation used
        threshold_prob: probability of threshold mutation
        is_class: is class mutation used
        class_prob: probability of class mutation
    """

    def __init__(self, is_feature: bool = True, feature_prob: float = 0.05,
                 is_threshold: bool = True, threshold_prob: float = 0.05,
                 is_class: bool = True, class_prob: float = 0.05, **kwargs):
        self.is_feature: bool = is_feature
        self.feature_prob: float = feature_prob
        self.is_threshold: bool = is_threshold
        self.threshold_prob: float = threshold_prob
        self.is_class: bool = is_class
        self.class_prob: float = class_prob

    def set_params(self, is_feature: bool = None, feature_prob: float = None,
                   is_threshold: bool = None, threshold_prob: float = None,
                   is_class: bool = None, class_prob: float = None):
        """
        Function to set new parameters for Mutator

        Arguments are the same as in __init__
        """
        if is_feature is not None:
            self.is_feature = is_feature
        if feature_prob is not None:
            self.feature_prob = feature_prob
        if is_threshold is not None:
            self.is_threshold = is_threshold
        if threshold_prob is not None:
            self.threshold_prob = threshold_prob
        if is_class is not None:
            self.is_class = is_class
        if class_prob is not None:
            self.class_prob = class_prob

    def mutate(self, forest: Forest):
        """
        It mutates all trees based on params

        Mutation of feature means changing feature in random decision node and
        mutate threshold in that node

        Mutation of threshold means changing threshold in random decision node

        Mutation of class means changing class in random leaf

        Args:
            forest: Container with all trees
        """
        trees_number: int = forest.current_trees
        if self.is_feature:
            tree_ids: np.array = self.get_random_trees(trees_number, self.feature_prob)
            for tree_id in tree_ids:
                tree: Tree = forest.trees[tree_id]
                tree.mutate_random_feature()
        if self.is_threshold:
            tree_ids: np.array = self.get_random_trees(trees_number, self.threshold_prob)
            for tree_id in tree_ids:
                tree: Tree = forest.trees[tree_id]
                tree.mutate_random_threshold()
        if self.is_class:
            tree_ids: np.array = self.get_random_trees(trees_number, self.class_prob)
            for tree_id in tree_ids:
                tree: Tree = forest.trees[tree_id]
                tree.mutate_random_class()
        return forest

    @staticmethod
    def get_random_trees(n_trees: int, probability: float) -> np.array:
        """
        Warning:
            It don't use normal probability of choosing each tree

            It brings random ceil(n_trees * probability) indices

        Args:
            n_trees: Number of trees in the forest
            probability: Probability of choosing each individual

        Return:
            np.array with indices
        """
        return np.random.choice(n_trees, math.ceil(n_trees * probability), replace=False)