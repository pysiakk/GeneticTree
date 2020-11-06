import numpy as np
import warnings

from aenum import Enum


def get_selected_indices_by_rank_selection(metrics, n_individuals):
    # in this type of selection we have to get best trees
    indices = np.argpartition(-metrics, n_individuals - 1)[:n_individuals]
    return indices


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
        assert n_elitism <= n_trees
        self.n_trees: int = n_trees
        self.selection_type: SelectionType = selection_type
        self.n_elitism: int = n_elitism

    def set_params(self,
                   n_trees: int = None,
                   selection_type: callable = None,
                   n_elitism: int = None,
                   **kwargs):
        """
        Function to set new parameters for Selector

        Arguments are the same as in __init__
        """
        if n_trees is not None:
            self.n_trees = n_trees
        if selection_type is not None:
            self.selection_type = selection_type
        if n_elitism is not None:
            assert n_elitism <= self.n_trees
            self.n_elitism = n_elitism

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
        elif n_elitism <= len(trees):
            n_elitism = len(trees)
        elite_indices = np.argpartition(-trees_metrics, n_elitism - 1)[:n_elitism]
        return list(np.take(np.array(trees), elite_indices))
