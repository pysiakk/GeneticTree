import numpy as np
import warnings

from aenum import Enum, extend_enum
from ..tree.tree import copy_tree


def metrics_greater_than_zero(metrics: np.array) -> np.array:
    """
    Changes metric array to be sure that each metric is greater than zero.
    Firstly it finds the minimum_value which is the value to replace metrics
    less than minimum_value. This value is try to be as small as possible but
    positive.

    Args:
         metrics: array with value of metric of each individual
    Returns:
         np.array: array with value of metric of each individual
    """
    minimum_value = np.min(np.abs(metrics))
    if minimum_value == 0:
        minimum_value = 10**-7
    return np.maximum(metrics, minimum_value)


def get_selected_indices_by_rank_selection(metrics: np.array, n_individuals: int, **kwargs) -> np.array:
    """
    Selects best (with the highest metric) n_individuals individuals

    Args:
         metrics: array with value of metric of each individual
         n_individuals: number of individuals to select
    Returns:
         np.array: array with indices of selected individuals (individuals are in random order)
    """
    if metrics.shape[0] >= n_individuals:
        indices = np.argpartition(-metrics, n_individuals - 1)[:n_individuals]
        return indices
    else:
        indices_recurrent = get_selected_indices_by_rank_selection(metrics, n_individuals - metrics.shape[0])
        return np.concatenate([np.arange(0, metrics.shape[0]), indices_recurrent])


def get_selected_indices_by_tournament_selection(metrics: np.array, n_individuals: int,
                                                 tournament_size: int = 3, **kwargs) -> np.array:
    """
    It simulates n_individuals tournaments
    At each tournament there are chosen random tournament_size individuals
    Then the best (with the highest metric) individual from tournament is selected

    Args:
         metrics: array with value of metric of each individual
         n_individuals: number of individuals to select
         tournament_size: Number of individuals in each tournament
    Returns:
         np.array: array with indices of selected individuals (individuals are in random order)
    """
    random_indices = np.empty([n_individuals, tournament_size], dtype=np.int)
    for i in range(tournament_size):
        random_indices[:, i] = np.random.randint(0, metrics.shape[0] - i, n_individuals)

    def tournament_selection(row):
        for j in range(1, row.shape[0]):
            row[j] += np.sum(row[:j] <= row[j])
        return row[np.argmax(metrics[row])]

    return np.apply_along_axis(tournament_selection, 1, random_indices)


def get_selected_indices_by_roulette_selection(metrics: np.array, n_individuals: int, **kwargs) -> np.array:
    """
    First it creates a circle on which every individual gets angle proportional
    to its metric (such that sum of all angles is all circle)
    Then algorithm spins the wheel n_individual times and selects pointed individual

    In practical implementation instead of angle there is a proportional part of
    [0, 1] section. And instead of spinning the wheel there are generated
    random numbers from [0, 1] section.

    Args:
         metrics: array with value of metric of each individual
         n_individuals: number of individuals to select
    Returns:
         np.array: array with indices of selected individuals (individuals are in random order)
    """
    metrics = metrics_greater_than_zero(metrics)

    metrics_summed = np.cumsum(metrics)
    metrics_summed = metrics_summed / metrics_summed[metrics.shape[0] - 1]

    # roulette numbers are from 0 to 1, because metrics_summed are rescaled to [0, 1]
    random_roulette_numbers = np.random.random(n_individuals)
    random_roulette_numbers = np.sort(random_roulette_numbers)

    selected_indices = np.empty(n_individuals, dtype=np.int)
    metric_index = 0
    roulette_index = 0
    selected_index = 0

    # the best theoretical computational complexity but in pure python
    while roulette_index != n_individuals:
        if random_roulette_numbers[roulette_index] <= metrics_summed[metric_index]:
            selected_indices[selected_index] = metric_index
            selected_index += 1
            roulette_index += 1
        else:
            metric_index += 1

    return selected_indices


def get_selected_indices_by_stochastic_uniform_selection(metrics: np.array, n_individuals: int, **kwargs) -> np.array:
    """
    First it creates a section on which every individual gets distance proportional
    to its metric, the individuals are positioned without intervals between.
    Then algorithm goes by the same distance on this section and selects pointed
    individuals.

    In practical implementation the section is scaled to [0, 1].
    Then the algorithm is going from one number to another by
    distance = 1 / n_individuals. And the algorithm starts from random number
    from [0, 1/n_individuals] section.

    Args:
         metrics: array with value of metric of each individual
         n_individuals: number of individuals to select
    Returns:
         np.array: array with indices of selected individuals (individuals are in random order)
    """
    metrics = metrics_greater_than_zero(metrics)

    metrics_summed = np.cumsum(metrics)
    metrics_summed = metrics_summed / metrics_summed[metrics.shape[0] - 1]

    distance = 1 / n_individuals
    random_number = np.random.random(1)[0] * distance

    selected_indices = np.empty(n_individuals, dtype=np.int)
    metric_index = 0
    selected_index = 0

    # the best theoretical computational complexity but in pure python
    while selected_index != n_individuals:
        if random_number <= metrics_summed[metric_index]:
            selected_indices[selected_index] = metric_index
            selected_index += 1
            random_number += distance
        else:
            metric_index += 1

    return selected_indices


class Selection(Enum):
    """
    Selection is enumerator with possible selections to use:
        Rank -- select best (based on metric) n trees
        Tournament -- n tournaments with k individuals - each tournament is won \
        by the best individual
        Roulette -- n spins of roulette on which each individual have angle \
        proportional to metric
        Stochastic Uniform -- n points belonging to individuals sections; points \
        are distant from each other the same distance; individuals sections are \
        proportional to metric

    To add new Selection execute code similar to:
    <code>
    def selection_function(metrics: np.array, n_individuals: int) -> np.array:
        # function that will get np array of trees metrics
        # and number of individuals to select
        # it returns np array with selected indices
        indices = ...
        return indices
    Selection.add_new("SelectionTypeName", selection_function)
    </code>
    Then you can use new selection type by passing in genetic tree
    Selection.SelectionTypeName
    """
    def __new__(cls, function, *args):
        obj = object.__new__(cls)
        obj._value_ = len(cls.__members__)
        obj.select = function
        return obj

    @staticmethod
    def add_new(name, function):
        extend_enum(Selection, name, function)

    # after each entry should be at least delimiter
    # (also can be more arguments which will be ignored)
    # this is needed because value is callable type
    Rank = get_selected_indices_by_rank_selection,
    Tournament = get_selected_indices_by_tournament_selection,
    Roulette = get_selected_indices_by_roulette_selection,
    StochasticUniform = get_selected_indices_by_stochastic_uniform_selection,


class Selector:
    """
    Selector is responsible for selecting best individuals from population

    Possible selection policies:
    - Rank (best n)
    - Tournament (n tournaments with k individuals - each tournament is won \
    by the best individual)
    - Roulette (n spins of roulette on which each individual have angle \
    proportional to metric)
    - Stochastic Uniform (n points belonging to individuals sections; points \
    are distant from each other the same distance; individuals sections are \
    proportional to metric

    There is also elitism, which allows to select best (in terms of trees
    metrics) n_elitism individuals

    Args:
        n_trees: number of trees to select
        selection: a selection policy how to select new individuals
        n_elitism: number of best trees to select unconditionally between 2 \
        iterations
    """

    def __init__(self,
                 n_trees: int = 400,
                 selection: Selection = Selection.StochasticUniform,
                 n_elitism: int = 3,
                 **kwargs):
        self.n_trees: int = self._check_n_trees(n_trees)
        self.selection: Selection = self._check_selection(selection)
        self.n_elitism: int = self._check_n_elitism(n_elitism)
        self._kwargs = kwargs

    def set_params(self,
                   n_trees: int = None,
                   selection: Selection = None,
                   n_elitism: int = None,
                   **kwargs):
        """
        Function to set new parameters for Selector

        Arguments are the same as in __init__
        """
        if n_trees is not None:
            self.n_trees = self._check_n_trees(n_trees)
        if selection is not None:
            self.selection = self._check_selection(selection)
        if n_elitism is not None:
            self.n_elitism = self._check_n_elitism(n_elitism)
        self._kwargs = dict(self._kwargs, **kwargs)

    @staticmethod
    def _check_n_trees(n_trees):
        if n_trees <= 0:
            warnings.warn(f"Try to set n_trees={n_trees}. Changed to n_trees=1, "
                          f"but try to set n_trees manually for value at least 20")
            n_trees = 1
        return n_trees

    @staticmethod
    def _check_selection(selection):
        # comparison of strings because after using Selection.add_new() Selection is reference to other class
        if str(type(selection)) == str(Selection):
            return selection
        else:
            raise TypeError(f"Passed selection={selection} with type {type(selection)}, "
                            f"Needed argument with type Selection")

    def _check_n_elitism(self, n_elitism):
        if n_elitism >= self.n_trees:
            n_elitism = self.n_trees
        if n_elitism <= 0:
            n_elitism = 0
        return n_elitism

    def select(self, trees, trees_metrics):
        """
        Function selects parents from population
        It selects self.n_trees trees
        It uses selection policy

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
        indices = self.selection.select(trees_metrics, self.n_trees, **self._kwargs)
        new_trees = self._get_new_trees_by_indices(trees, indices)
        return new_trees

    @staticmethod
    def _get_new_trees_by_indices(trees, indices):
        sorted_indices = np.sort(indices)

        # first get all unique indices
        unique_indices = np.unique(indices)
        new_trees = list(np.take(np.array(trees), unique_indices))

        # remove one copy of each index in indices
        uniques_to_remove = np.searchsorted(sorted_indices, unique_indices)
        removed = np.delete(sorted_indices, uniques_to_remove)

        # for each existing index copy tree and add to new_trees list
        # it is needed to not have two references for the same tree
        for index in removed:
            new_trees.append(copy_tree(trees[index]))

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
