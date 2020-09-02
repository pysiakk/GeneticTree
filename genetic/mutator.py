from tree.forest import Forest
from tree.tree import Tree
import math
import numpy as np


class Mutator:
    """
    Class responsible for mutation of individuals
    """

    def __init__(self, mutate_features: bool = True, change_feature: float = 0.05,
                 mutate_thresholds: bool = True, change_threshold: float = 0.05,
                 mutate_classes: bool = True, change_class: float = 0.05, **kwargs):
        self.mutate_features: bool = mutate_features
        self.change_feature: float = change_feature
        self.mutate_thresholds: bool = mutate_thresholds
        self.change_threshold: float = change_threshold
        self.mutate_classes: bool = mutate_classes
        self.change_class: float = change_class

    def set_params(self, mutate_features: bool = None, change_feature: float = None,
                   mutate_thresholds: bool = None, change_threshold: float = None,
                   mutate_classes: bool = None, change_class: float = None):
        if mutate_features is not None:
            self.mutate_features = mutate_features
        if change_feature is not None:
            self.change_feature = change_feature
        if mutate_thresholds is not None:
            self.mutate_thresholds = mutate_thresholds
        if change_threshold is not None:
            self.change_threshold = change_threshold
        if mutate_classes is not None:
            self.mutate_classes = mutate_classes
        if change_class is not None:
            self.change_class = change_class

    def mutate(self, forest: Forest):
        trees_number: int = forest.current_trees
        if self.mutate_features:
            tree_ids: np.array = self.get_random_trees(trees_number, self.change_feature)
            for tree_id in tree_ids:
                tree: Tree = forest.trees[tree_id]
                tree.mutate_random_feature()
        if self.mutate_thresholds:
            tree_ids: np.array = self.get_random_trees(trees_number, self.change_threshold)
            for tree_id in tree_ids:
                tree: Tree = forest.trees[tree_id]
                tree.mutate_random_threshold()
        if self.mutate_classes:
            tree_ids: np.array = self.get_random_trees(trees_number, self.change_class)
            for tree_id in tree_ids:
                tree: Tree = forest.trees[tree_id]
                tree.mutate_random_class()

    @staticmethod
    def get_random_trees(trees_number: int, probability: float):
        return np.random.choice(trees_number, math.ceil(trees_number * probability), replace=False)