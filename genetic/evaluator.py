from tree.evaluation import get_accuracies, get_proper_classified, get_trees_sizes
import numpy as np

from enum import Enum, auto


class Metric(Enum):
    Accuracy = auto()
    AccuracyBySize = auto()


class Evaluator:
    """
    Evaluator is responsible for evaluating each individuals' score
    It evaluates each individual and give them a score

    There are plenty of possible metrics:
    - accuracy
    - accuracy with size

    Args:
        n_trees: number of trees to select
        metric: a metric used to evaluate single tree
        size_coef: coefficient inside AccuracyBySize Metric
    """

    def __init__(self,
                 n_trees: int = 200,
                 metric: Metric = Metric.AccuracyBySize,
                 size_coef: int = 1000,
                 **kwargs):
        self.n_trees: int = n_trees
        self.metric: Metric = metric
        self.size_coef = size_coef

    def set_params(self,
                   n_trees: int = None,
                   metric: Metric = None,
                   size_coef: int = None,
                   **kwargs):
        """
        Function to set new parameters for Selector

        Arguments are the same as in __init__
        """
        if n_trees is not None:
            self.n_trees = n_trees
        if metric is not None:
            self.metric = metric
        if size_coef is not None:
            self.size_coef = size_coef

    def get_best_tree_index(self, trees) -> int:
        """
        Args:
            forest: Container with all trees

        Returns:
            Index of best tree inside forest
        """
        trees_metric = self.evaluate(trees)
        best_index = np.argmax(trees_metric)
        return best_index

    def evaluate(self, trees):
        """
        Function evaluates each tree's metric inside forest
        The metrics are stored then in field Selector.tree_metric

        Args:
            forest: Container with all trees
        """
        if self.metric == Metric.Accuracy:
            trees_metric = np.array(get_accuracies(trees))
        elif self.metric == Metric.AccuracyBySize:
            acc = np.array(get_accuracies(trees))
            size = np.array(get_trees_sizes(trees))
            trees_metric = acc ** 2 * self.size_coef / (self.size_coef + size ** 2)
        else:
            raise ValueError(f"The metric {self.metric} not exist")
        return trees_metric
