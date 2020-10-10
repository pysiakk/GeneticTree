from tree.forest import Forest
from tree.tree import Tree
import numpy as np

from enum import Enum, auto


class SelectionType(Enum):
    RankSelection = auto()


class Metric(Enum):
    Accuracy = auto()
    AccuracyBySize = auto()


class Selector:
    """
    Selector is responsible for selecting best individuals from population

    First of all it evaluates each individual and give them a score
    Then it selects best population

    There are plenty of possible metrics:
    - accuracy

    Possible selection policies:
    - best n

    There is also elitarysm, which means that top k individuals are selected
    before selection policy is used

    Args:
        n_trees: number of trees to select
        selection_type: a selection policy how to select new individuals
        metric: a metric used to evaluate single tree
        size_coef: coefficient inside AccuracyBySize Metric
        elitarysm: number of best trees to select before selecting other trees by selection_type policy
    """

    def __init__(self,
                 n_trees: int = 200,
                 selection_type: SelectionType = SelectionType.RankSelection,
                 metric: Metric = Metric.AccuracyBySize,
                 size_coef: int = 1000,
                 elitarysm: int = 5,
                 **kwargs):
        assert elitarysm <= n_trees
        self.n_trees: int = n_trees
        self.selection_type: SelectionType = selection_type
        self.metric: Metric = metric
        self.size_coef = size_coef
        self.n_elitarysm: int = elitarysm

    def set_params(self, n_trees: int = None, selection_type: SelectionType = None,
                   metric: Metric = None, elitarysm: int = None):
        """
        Function to set new parameters for Selector

        Arguments are the same as in __init__
        """
        if n_trees is not None:
            self.n_trees = n_trees
        if selection_type is not None:
            self.selection_type = selection_type
        if metric is not None:
            self.metric = metric
        if elitarysm is not None:
            assert elitarysm <= self.n_trees
            self.n_elitarysm = elitarysm

    def get_best_tree_index(self, forest: Forest) -> int:
        """
        Args:
            forest: Container with all trees

        Returns:
            Index of best tree inside forest
        """
        self.__evaluate__(forest)
        best_index = np.argmax(self.trees_metric)
        return best_index

    def select(self, forest: Forest):
        """
        Function that changes trees np.array inside forest to contain only
        trees selected by usage of SelectionType

        Args:
            forest: Container with all trees
        """
        self.__evaluate__(forest)
        self.__leave_best_population__(forest)

    def __evaluate__(self, forest: Forest):
        """
        Function evaluates each tree's metric inside forest
        The metrics are stored then in field Selector.tree_metric

        Args:
            forest: Container with all trees
        """
        if self.metric == Metric.Accuracy:
            self.trees_metric = np.array(forest.get_accuracies())
        if self.metric == Metric.AccuracyBySize:
            acc = np.array(forest.get_accuracies())
            size = np.array(forest.get_trees_sizes())
            self.trees_metric = acc ** 2 * self.size_coef / (self.size_coef + size ** 2)

    def __leave_best_population__(self, forest: Forest):
        """
        Function leaves population (selected by SelectionType) inside forest
        It need to metric of all trees be evaluated before

        Args:
            forest: Container with all trees
        """
        # selected_indices contains indices of trees selected by elitarysm and SelectionType
        self.selected_indices = np.zeros(self.n_trees)
        self.__set_elite_indices__()
        self.__set_selected_indices__()

        max_n_trees = len(forest.trees)
        new_trees = np.empty(max_n_trees, Tree)
        new_trees[:self.n_trees] = np.take(forest.trees, self.selected_population)
        forest.trees = new_trees
        forest.current_trees = self.n_trees

    def __set_elite_indices__(self):
        """
        Function sets indices of best n_elitarysm trees
        """
        if self.n_elitarysm <= 0:
            return
        elite_indices = np.argpartition(-self.trees_metric, self.n_elitarysm - 1)[:self.n_elitarysm]
        self.selected_indices[:self.n_elitarysm] = elite_indices

    def __set_selected_indices__(self):
        """
        Function sets indices of trees selected by SelectionType
        """
        if self.selection_type == SelectionType.RankSelection:
            self.__set_selected_indices_by_rank_selection()

    def __set_selected_indices_by_rank_selection(self):
        # in this type of selection we have to get best trees
        # so this is the same as elitarysm of size n_trees
        # so it can be done as returning best n_trees trees
        indices = np.argpartition(-self.trees_metric, self.n_trees-1)[:self.n_trees]
        self.selected_population = indices
        # in other SelectionType's it should be done as:
        # self.selected_population[self.elitarysm:] = indices
        # and indices array should be the size of self.n_trees - self.elitarysm
