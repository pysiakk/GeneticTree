import numpy as np
import warnings

from aenum import Enum


def get_selected_indices_by_rank_selection(metrics, n_individuals):
    # in this type of selection we have to get best trees
    indices = np.argpartition(-metrics, n_individuals - 1)[:n_individuals]
    return indices


def get_selected_indices_by_tournament_selection(metrics, n_individuals, tournament_size=3):
    random_indices = np.empty([n_individuals, tournament_size], dtype=np.int)
    for i in range(tournament_size):
        random_indices[:, i] = np.random.randint(0, metrics.shape[0] - i, n_individuals)

    def tournament_selection(row):
        for j in range(1, row.shape[0]):
            row[j] += np.sum(row[:j] <= row[j])
        return row[np.argmax(metrics[row])]

    return np.apply_along_axis(tournament_selection, 1, random_indices)


class SelectionType(Enum):
    """
    SelectionType is enumerator with possible selections to use:
        RankSelection -- select best (based on metric) n trees

    To add new SelectionType execute code similar to:
    <code>
    from aenum import extend_enum
    name = "SelectionTypeName" # string with name of new selection type
    def selection_function(metrics, n_individuals):
        # function that will get np array of trees metrics
        # and number of individuals to select
        # it returns np array with selected indices
        indices = ...
        return indices
    extend_enum(SelectionType, name, selection_function)
    </code>
    Then you can use new selection type by passing in genetic tree
    SelectionType.SelectionTypeName
    """
    def __new__(cls, function, *args):
        obj = object.__new__(cls)
        obj._value_ = len(cls.__members__)
        obj.select = function
        return obj

    # after each entry should be at least delimiter
    # (also can be more arguments which will be ignored)
    # this is needed because value is callable type
    RankSelection = get_selected_indices_by_rank_selection,
    TournamentSelection = get_selected_indices_by_tournament_selection,


class Selector:
    """
    Selector is responsible for selecting best individuals from population

    Possible selection policies:
    - rankSelection (best n)

    There is also elitism, which allows to select best (in terms of trees
    metrics) n_elitism individuals

    Args:
        n_trees: number of trees to select
        selection_type: a selection policy how to select new individuals
        n_elitism: number of best trees to select unconditionally between 2 \
        iterations
    """

    def __init__(self,
                 n_trees: int = 200,
                 selection_type: SelectionType = SelectionType.RankSelection,
                 n_elitism: int = 5,
                 **kwargs):
        self.n_trees: int = self._check_n_trees_(n_trees)
        self.selection_type: SelectionType = self._check_selection_type_(selection_type)
        self.n_elitism: int = self._check_n_elitism_(n_elitism)

    def set_params(self,
                   n_trees: int = None,
                   selection_type: SelectionType = None,
                   n_elitism: int = None,
                   **kwargs):
        """
        Function to set new parameters for Selector

        Arguments are the same as in __init__
        """
        if n_trees is not None:
            self.n_trees = self._check_n_trees_(n_trees)
        if selection_type is not None:
            self.selection_type = self._check_selection_type_(selection_type)
        if n_elitism is not None:
            self.n_elitism = self._check_n_elitism_(n_elitism)

    @staticmethod
    def _check_n_trees_(n_trees):
        if n_trees <= 0:
            warnings.warn(f"Try to set n_trees={n_trees}. Changed to n_trees=1, "
                          f"but try to set n_trees manually for value at least 20")
            n_trees = 1
        return n_trees

    @staticmethod
    def _check_selection_type_(selection_type):
        if isinstance(selection_type, SelectionType):
            return selection_type
        else:
            raise TypeError(f"Passed selection_type={selection_type} with type {type(selection_type)}, "
                            f"Needed argument with type SelectionType")

    def _check_n_elitism_(self, n_elitism):
        if n_elitism >= self.n_trees:
            n_elitism = self.n_trees
        if n_elitism <= 0:
            n_elitism = 0
        return n_elitism

    def select(self, trees, trees_metrics):
        """
        Function selects best parents from population

        Args:
            trees: List with all trees
            trees_metrics: Metric of each tree
        """
        if trees_metrics.shape[0] < self.n_trees:
            warnings.warn(f"There are {trees_metrics.shape[0]} trees but it has "
                          f"to be selected {self.n_trees}. If algorithm will "
                          f"throw error that there arent any trees try to change "
                          f"parameters so that on each iteration could be "
                          f"created more tree. For example dont replace parents "
                          f"by offspring or set bigger mutation or crossing "
                          f"probability with do not replacing parents.",
                          UserWarning)
        indices = self.selection_type.select(trees_metrics, self.n_trees)
        # TODO take only once each index, then if index is repeated make a copy of tree
        new_trees = list(np.take(np.array(trees), indices))
        return new_trees

    def get_elite_population(self, trees, trees_metrics):
        """
        Function to select best n_elitism trees

        Args:
            trees: List with all trees
            trees_metrics: Metric of each tree
        """
        n_elitism = self.n_elitism
        if n_elitism <= 0:
            return []
        elif n_elitism >= len(trees):
            n_elitism = len(trees)
        elite_indices = np.argpartition(-trees_metrics, n_elitism - 1)[:n_elitism]
        return list(np.take(np.array(trees), elite_indices))
