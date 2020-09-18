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
            self.elitarysm = elitarysm

    def select(self, forest: Forest):
        self.__evaluate__(forest)
        self.__leave_best_population__(forest)

    def __evaluate__(self, forest: Forest):
        proper_classified = np.empty(len(forest.trees))
        for i in range(forest.current_trees):
            proper_classified[i] = self.__evaluate_single_tree__(forest.trees[i])
        self.proper_classified = proper_classified

    def __leave_best_population__(self, forest: Forest):
        #TODO
        pass

    def __evaluate_single_tree__(self, tree: Tree):
        observations: dict = tree.observations
        proper_classified: int = 0
        for k, val in observations.items():
            for item in val:
                if item.proper_class == item.current_class:
                    proper_classified += 1
        return proper_classified
