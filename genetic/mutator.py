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
        is_node: is node mutation used \
        (if node is leaf mutate class, otherwise mutate feature)
        node_prob: probability of node mutation
        is_class_or_threshold: is class_or_threshold mutation used \
        (if node is leaf mutate class, otherwise mutate threshold)
        class_or_threshold_prob: probability of class_or_threshold mutation
    """

    def __init__(self,
                 is_feature: bool = False, feature_prob: float = 0.005,
                 is_threshold: bool = False, threshold_prob: float = 0.005,
                 is_class: bool = False, class_prob: float = 0.005,
                 is_node: bool = False, node_prob: float = 0.005,
                 is_class_or_threshold: bool = True, class_or_threshold_prob: float = 0.005,
                 **kwargs):
        self.is_feature: bool = is_feature
        self.feature_prob: float = feature_prob
        self.is_threshold: bool = is_threshold
        self.threshold_prob: float = threshold_prob
        self.is_class: bool = is_class
        self.class_prob: float = class_prob
        self.is_node: bool = is_node
        self.node_prob: float = node_prob
        self.is_class_or_threshold: bool = is_class_or_threshold
        self.class_or_threshold_prob: float = class_or_threshold_prob

    def set_params(self,
                   is_feature: bool = None, feature_prob: float = None,
                   is_threshold: bool = None, threshold_prob: float = None,
                   is_class: bool = None, class_prob: float = None,
                   is_node: bool = None, node_prob: float = None,
                   is_class_or_threshold: bool = None, class_or_threshold_prob: float = None,
                   **kwargs):
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
        if is_node is not None:
            self.is_node = is_node
        if node_prob is not None:
            self.node_prob = node_prob
        if is_class_or_threshold is not None:
            self.is_class_or_threshold = is_class_or_threshold
        if class_or_threshold_prob is not None:
            self.class_or_threshold_prob = class_or_threshold_prob

    def mutate(self, trees):
        """
        It mutates all trees based on params

        Mutation of feature means changing feature in random decision node and
        mutate threshold in that node

        Mutation of threshold means changing threshold in random decision node

        Mutation of class means changing class in random leaf

        Mutation of node means mutating feature for decision node or class for
        leaf

        Mutation of class_or_threshold means mutating threshold for decision
        node or class for leaf

        Args:
            forest: Container with all trees

        Returns:
            forest: Container with mutated trees
        """
        if self.is_feature:
            self.mutate_one(trees, Tree.mutate_random_feature, self.feature_prob)
        if self.is_threshold:
            self.mutate_one(trees, Tree.mutate_random_threshold, self.threshold_prob)
        if self.is_class:
            self.mutate_one(trees, Tree.mutate_random_class, self.class_prob)
        if self.is_node:
            self.mutate_one(trees, Tree.mutate_random_node, self.node_prob)
        if self.is_class_or_threshold:
            self.mutate_one(trees, Tree.mutate_random_class_or_threshold, self.class_or_threshold_prob)

    @staticmethod
    def mutate_one(trees, function: callable, prob: float):
        """
        It mutate all trees by function with prob probability

        Args:
            forest: Container with all trees
            function: A tree function which perform mutation on tree
            prob: The probability that each tree will be mutated

        Returns:
            forest: Container with mutated trees
        """
        trees_number: int = len(trees)
        tree_ids: np.array = Mutator.get_random_trees(trees_number, prob)
        for tree_id in tree_ids:
            tree: Tree = trees[tree_id]
            function(tree)

    @staticmethod
    def get_random_trees(n_trees: int, probability: float) -> np.array:
        """
        Warning:
            It don't use normal probability of choosing each tree

            It brings random ceil(n_trees * probability) indices

        Args:
            n_trees: Number of trees in the forest
            probability: Probability of choosing each individual

        Returns:
            np.array with indices
        """
        return np.random.choice(n_trees, math.ceil(n_trees * probability), replace=False)