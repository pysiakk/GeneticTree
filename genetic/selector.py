import numpy as np

from enum import Enum, auto


class SelectionType(Enum):
    RankSelection = auto()


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
                 elitarysm: int = 5,
                 **kwargs):
        assert elitarysm <= n_trees
        self.n_trees: int = n_trees
        self.selection_type: SelectionType = selection_type
        self.n_elitarysm: int = elitarysm

    def set_params(self,
                   n_trees: int = None,
                   selection_type: SelectionType = None,
                   elitarysm: int = None,
                   **kwargs):
        """
        Function to set new parameters for Selector

        Arguments are the same as in __init__
        """
        if n_trees is not None:
            self.n_trees = n_trees
        if selection_type is not None:
            self.selection_type = selection_type
        if elitarysm is not None:
            assert elitarysm <= self.n_trees
            self.n_elitarysm = elitarysm

    def select(self, trees, trees_metric):
        """
        Function leaves population (selected by SelectionType) inside forest
        It need to metric of all trees be evaluated before

        Args:
            forest: Container with all trees
            trees_metric: Metric of each tree
        """
        # selected_indices contains indices of trees selected by elitarysm and SelectionType
        self.trees_metric = trees_metric
        self.selected_indices = np.zeros(self.n_trees)
        self.__set_elite_indices__()
        self.__set_selected_indices__()

        new_trees = list(np.take(np.array(trees), self.selected_population))
        return new_trees

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
