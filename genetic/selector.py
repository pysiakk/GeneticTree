from tree.forest import Forest
from tree.tree import Tree
import numpy as np

from enum import Enum, auto


class SelectionType(Enum):
    RankSelection = auto()


class Metric(Enum):
    Accuracy = auto()


class Selector:
    """
    Class responsible for selecting best individuals from population
    First of all it evaluates each individual and give them a score
    Then it selects best population

    There are plenty of possible metrics:
    - accuracy

    Possible selection policies:
    - best n

    There is also elitarysm, which means that top k individuals are selected
    before selection policy is used
    """

    def __init__(self, n_trees: int = 1000, selection_type: SelectionType = SelectionType.RankSelection,
                 metric: Metric = Metric.Accuracy, elitarysm: int = 5, **kwargs):
        assert elitarysm <= n_trees
        self.n_trees: int = n_trees
        self.selection_type: SelectionType = selection_type
        self.metric: Metric = metric
        self.elitarysm: int = elitarysm
        self.n_observations = None

    def set_n_observations(self, n_observations: int):
        self.n_observations = n_observations

    def set_params(self, n_trees: int = None, selection_type: SelectionType = None,
                   metric: Metric = None, elitarysm: int = None):
        if n_trees is not None:
            self.n_trees = n_trees
        if selection_type is not None:
            self.selection_type = selection_type
        if metric is not None:
            self.metric = metric
        if elitarysm is not None:
            assert elitarysm <= self.n_trees
            self.elitarysm = elitarysm

    def get_best_tree_index(self, forest: Forest) -> int:
        self.__evaluate__(forest)
        best_index = np.argmax(self.trees_metric)
        return best_index

    def select(self, forest: Forest):
        self.__evaluate__(forest)
        self.__leave_best_population__(forest)

    def __evaluate__(self, forest: Forest):
        proper_classified = np.empty(forest.current_trees)
        for i in range(forest.current_trees):
            proper_classified[i] = self.__evaluate_single_tree__(forest.trees[i], forest.X)
        self.trees_metric = self.__get_trees_by_metric__(proper_classified)

    def __leave_best_population__(self, forest: Forest):
        self.selected_population = np.zeros(self.n_trees)
        self.__set_elite_indices__()
        self.__set_selected_indices__()

        max_trees = len(forest.trees)
        new_trees = np.empty(max_trees, Tree)
        new_trees[:self.n_trees] = np.take(forest.trees, self.selected_population)
        forest.trees = new_trees
        forest.current_trees = self.n_trees

    """
    set indices of the best trees
    """
    def __set_elite_indices__(self):
        if self.elitarysm <= 0:
            return []
        elite_indices = np.argpartition(-self.trees_metric, self.elitarysm-1)[:self.elitarysm]
        self.selected_population[:self.elitarysm] = elite_indices

    """
    set indices of trees selected by SelectionType
    """
    def __set_selected_indices__(self):
        if self.selection_type == SelectionType.RankSelection:
            self.__set_selected_indices_by_rank_selection()

    def __set_selected_indices_by_rank_selection(self):
        # in this type of selection we have to get best trees
        # so this is the same as elitarysm of size n_trees
        # so it can be done as returning best n_trees trees
        indices = np.argpartition(-self.trees_metric, self.n_trees-1)[:self.n_trees]
        self.selected_population = indices
        # normally it should be done as:
        # self.selected_population[self.elitarysm:] = indices
        # and indices array should be the size of self.n_trees - self.elitarysm

    def __evaluate_single_tree__(self, tree: Tree, X):
        tree.assign_all_not_registered_observations(X)
        observations: dict = tree.observations
        proper_classified: int = 0
        for k, val in observations.items():
            for item in val:
                if item.proper_class == item.current_class:
                    proper_classified += 1
        return proper_classified

    """
    :returns np.array with indexes as forest.trees with numbers proportional to metric
    the bigger number - the better tree is
    """
    def __get_trees_by_metric__(self, proper_classified):
        if self.metric == Metric.Accuracy:
            return proper_classified
