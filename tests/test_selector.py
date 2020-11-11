from genetic.selector import SelectionType, Selector
from genetic.selector import get_selected_indices_by_rank_selection
from genetic.selector import get_selected_indices_by_tournament_selection
from genetic.selector import get_selected_indices_by_roulette_selection
from genetic.selector import get_selected_indices_by_stochastic_uniform_selection

from sklearn.utils._testing import assert_array_equal

import numpy as np
import pytest
import copy


# ==============================================================================
# Selection functions
# ==============================================================================
@pytest.fixture
def metrics():
    return np.array([18,  5, 12, 11,  20, 17,  8, 10,  6,  5,  3,  2, 14,  1,  4])


# +++++++++++++++
# Rank selection
# +++++++++++++++
@pytest.mark.parametrize("n_individuals",
                         [i for i in range(3, 18, 3)])
def test_rank_selection(metrics, n_individuals):
    selected_indices = get_selected_indices_by_rank_selection(metrics, n_individuals)
    proper_indices = np.argsort(-metrics)[:n_individuals]
    assert_array_equal(np.sort(selected_indices), np.sort(proper_indices))


# +++++++++++++++
# Tournament selection
# +++++++++++++++
def tournament_selection(metrics, n_individuals, tournament_size, proper_array):
    np.random.seed(123)
    selected_indices = get_selected_indices_by_tournament_selection(metrics, n_individuals, tournament_size)
    assert_array_equal(selected_indices, proper_array)


@pytest.mark.parametrize("n_individuals, tournament_size",
                         [(10, 3), (10, 5)])
def test_tournament_selection_reproducible_results(metrics, n_individuals, tournament_size):
    np.random.seed(123)
    result1 = get_selected_indices_by_tournament_selection(metrics, n_individuals, tournament_size)
    np.random.seed(123)
    result2 = get_selected_indices_by_tournament_selection(metrics, n_individuals, tournament_size)
    assert_array_equal(result1, result2)


def test_tournament_selection1(metrics):
    tournament_selection(metrics, 10, 3, np.array([3, 4, 0, 0, 0, 2, 6, 0, 0, 4]))


def test_tournament_selection2(metrics):
    tournament_selection(metrics, 10, 5, np.array([3, 4, 0, 0, 0, 2, 3, 0, 0, 4]))


def test_tournament_selection_with_size_1(metrics):
    np.random.seed(123)
    proper_array = np.random.randint(0, 15, 10)
    tournament_selection(metrics, 10, 1, proper_array)


# +++++++++++++++
# Roulette selection
# +++++++++++++++

@pytest.mark.parametrize("n_individuals", [5, 10])
def test_roulette_selection_reproducible_results(metrics, n_individuals):
    np.random.seed(123)
    result1 = get_selected_indices_by_roulette_selection(metrics, n_individuals)
    np.random.seed(123)
    result2 = get_selected_indices_by_roulette_selection(metrics, n_individuals)
    assert_array_equal(result1, result2)


@pytest.mark.parametrize("n_individuals", [2, 5, 10])
def test_roulette_selection(metrics, n_individuals):
    np.random.seed(123)
    selected_indices = get_selected_indices_by_roulette_selection(metrics, n_individuals)
    np.random.seed(123)
    indices = np.sort(np.random.random(n_individuals))
    metrics_summed = np.cumsum(metrics)
    metrics_summed = metrics_summed / metrics_summed[14]
    selected_indices_manually = np.empty(n_individuals, np.int)
    for i in range(indices.shape[0]):
        below = copy.deepcopy(metrics_summed)
        below[1:] = metrics_summed[:14]
        below[0] = 0
        above = metrics_summed
        selected_indices_manually[i] = np.argmax(np.logical_and(below <= indices[i], indices[i] < above))
    assert_array_equal(selected_indices, selected_indices_manually)


# +++++++++++++++
# Stochastic uniform selection
# +++++++++++++++

@pytest.mark.parametrize("n_individuals", [2, 5, 10])
def test_roulette_selection(metrics, n_individuals):
    np.random.seed(123)
    selected_indices = get_selected_indices_by_stochastic_uniform_selection(metrics, n_individuals)

    np.random.seed(123)
    distance = 1 / n_individuals
    first_number = np.random.random(1)[0] * distance
    indices = np.arange(0, n_individuals) * distance + first_number

    metrics_summed = np.cumsum(metrics)
    metrics_summed = metrics_summed / metrics_summed[14]

    selected_indices_manually = np.empty(n_individuals, np.int)
    for i in range(indices.shape[0]):
        below = copy.deepcopy(metrics_summed)
        below[1:] = metrics_summed[:14]
        below[0] = 0
        above = metrics_summed
        selected_indices_manually[i] = np.argmax(np.logical_and(below <= indices[i], indices[i] < above))
    assert_array_equal(selected_indices, selected_indices_manually)


# ==============================================================================
# Selector
# ==============================================================================

# Base selector
@pytest.fixture
def selector():
    return Selector(10, SelectionType.Rank, 3)


@pytest.fixture
def trees():
    return np.random.random(15)


# +++++++++++++++
# Set params
# +++++++++++++++

@pytest.mark.parametrize("n_trees", [1, 5, 1000])
def test_set_n_trees(selector, n_trees):
    selector.set_params(n_trees=n_trees)
    assert selector.n_trees == n_trees


@pytest.mark.parametrize("n_trees", [-10, 0])
def test_set_n_trees_below_one(selector, n_trees):
    selector.set_params(n_trees=n_trees)
    assert selector.n_trees == 1


@pytest.mark.parametrize("selection_type", [SelectionType.Rank,
                                            SelectionType.Tournament])
def test_set_selection_type(selector, selection_type):
    selector.set_params(selection_type=selection_type)
    assert selector.selection_type == selection_type


@pytest.mark.parametrize("selection_type", ["some_string"])
def test_set_selection_type_with_wrong_type(selector, selection_type):
    with pytest.raises(TypeError):
        selector.set_params(selection_type=selection_type)


@pytest.mark.parametrize("n_elitism", [1, 5])
def test_set_n_elitism(selector, n_elitism):
    selector.set_params(n_elitism=n_elitism)
    assert selector.n_elitism == n_elitism


def test_set_n_elitism_below_zero(selector):
    selector.set_params(n_elitism=-10)
    assert selector.n_elitism == 0


def test_set_n_elitism_above_n_trees(selector):
    selector.set_params(n_elitism=15)
    assert selector.n_elitism == 10


# +++++++++++++++
# Elitism
# +++++++++++++++

def test_elitism(selector, metrics, trees):
    elite_population = selector.get_elite_population(trees, metrics)
    elite_indices = np.argsort(-metrics)[:selector.n_elitism]
    assert_array_equal(np.sort(trees[elite_indices]), np.sort(elite_population))


def test_elitism_below_zero(selector, metrics, trees):
    selector.set_params(n_elitism=-10)
    assert_array_equal(selector.get_elite_population(trees, metrics), [])


@pytest.mark.parametrize("trees_len", [3, 7, 9])
def test_elitism_above_trees_len(selector, metrics, trees, trees_len):
    selector.set_params(n_elitism=10)
    assert selector.get_elite_population(trees[:trees_len], metrics[:trees_len]).__len__() == trees_len


# +++++++++++++++
# SelectionTypes
# +++++++++++++++

@pytest.mark.parametrize("selection_type", [SelectionType.Rank,
                                            SelectionType.Tournament,
                                            SelectionType.Roulette,
                                            SelectionType.StochasticUniform
                                            ])
def test_calling_proper_selection_type(selector, metrics, trees, selection_type):
    selector.set_params(selection_type=selection_type)
    np.random.seed(123)
    trees_by_selector = selector.select(trees, metrics)
    np.random.seed(123)
    indices_by_selection_type = selection_type.select(metrics, selector.n_trees)
    assert_array_equal(trees[indices_by_selection_type], trees_by_selector)


# ==============================================================================
# SelectionType
# ==============================================================================

def random_function():
    """Never can't be sure that function is not random"""
    return 9


def test_add_new_selection_type():
    SelectionType.add_new("MyNewName", random_function)
    assert SelectionType.MyNewName.select() == 9
