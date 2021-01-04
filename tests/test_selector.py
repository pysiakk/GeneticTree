from tests.utils_testing import *


# ==============================================================================
# Selection functions
# ==============================================================================
@pytest.fixture
def metrics():
    return np.array([18,  5, 12, 11,  20, 17,  8, 10,  6,  5,  3,  2, 14,  1,  4])


@pytest.mark.parametrize("metrics",
                         [[0, 0], [0, 1], [-1, 1], [-2, -5], [2, 3, 4]])
def test_metrics_greater_than_zero(metrics):
    metrics = metrics_greater_than_zero(metrics)
    for i in range(len(metrics)):
        assert metrics[i] > 0


@pytest.mark.parametrize("metrics",
                         [[2, 3, 4], [1, 7, 2, 4, 8]])
def test_metrics_greater_than_zero_dont_change_positive_metrics(metrics):
    metrics_changed = metrics_greater_than_zero(metrics)
    assert_array_equal(metrics, metrics_changed)


# +++++++++++++++
# Rank selection
# +++++++++++++++
@pytest.mark.parametrize("n_individuals",
                         [i for i in range(3, 18, 3)])
def test_rank_selection(metrics, n_individuals):
    selected_indices = get_selected_indices_by_rank_selection(metrics, n_individuals)
    proper_indices = np.argsort(-metrics)[:n_individuals]
    assert_array_equal(np.sort(selected_indices), np.sort(proper_indices))


@pytest.mark.parametrize("n_individuals",
                         [20, 40, 45, 46])
def test_rank_selection_with_n_individuals_bigger_than_metrics_shape(metrics, n_individuals):
    np.random.seed(123)
    selected_indices = get_selected_indices_by_rank_selection(metrics, n_individuals)
    repeats = n_individuals // metrics.shape[0]
    by_repeats = np.repeat(np.arange(0, metrics.shape[0]), repeats)
    np.random.seed(123)
    individuals_left = n_individuals - repeats * metrics.shape[0]
    by_ranking = np.argpartition(-metrics, individuals_left - 1)[:individuals_left]
    proper_indices = np.concatenate([by_repeats, by_ranking])
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
    return Selector(10, Selection.Rank, 3)


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


@pytest.mark.parametrize("selection", [Selection.Rank,
                                            Selection.Tournament])
def test_set_selection(selector, selection):
    selector.set_params(selection=selection)
    assert selector.selection == selection


@pytest.mark.parametrize("selection", ["some_string"])
def test_set_selection_with_wrong_type(selector, selection):
    with pytest.raises(TypeError):
        selector.set_params(selection=selection)


def test_set_new_selection(selector):
    Selection.add_new("NewSelector", lambda x: x)
    selector.set_params(selection=Selection.NewSelector)
    assert str(type(selector.selection)) == str(Selection)


def test_set_params_kwargs(selector):
    selector.set_params(a=2)
    assert selector._kwargs['a'] == 2
    selector.set_params(a=3)
    assert selector._kwargs['a'] == 3


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
# select
# +++++++++++++++

@pytest.fixture
def real_trees():
    trees = []
    for i in range(3):
        trees.append(build_trees(5, 1)[0])
    return trees


@pytest.fixture
def real_metrics():
    return np.array([1, 3, 2])


def test_copying_trees(selector, real_trees, real_metrics):
    # should select last tree 3 times but also should copy this tree
    selector.set_params(3, selection=Selection.Tournament)
    selected_trees = selector.select(real_trees, real_metrics)
    assert id(selected_trees[0]) != id(selected_trees[1]) != id(selected_trees[2]) != id(selected_trees[0])
    assert_array_equal(selected_trees[0].feature, selected_trees[1].feature)
    assert_array_equal(selected_trees[2].feature, selected_trees[1].feature)
    assert_array_equal(selected_trees[0].feature, selected_trees[2].feature)
    assert id(selected_trees[0]) == id(real_trees[1])


# +++++++++++++++
# SelectionTypes
# +++++++++++++++

@pytest.mark.parametrize("selection", [Selection.Rank,
                                            Selection.Tournament,
                                            Selection.Roulette,
                                            Selection.StochasticUniform
                                            ])
def test_calling_proper_selection(selector, real_metrics, real_trees, selection):
    selector.set_params(selection=selection, n_trees=2)
    np.random.seed(123)
    trees_by_selector = selector.select(real_trees, real_metrics)
    np.random.seed(123)
    indices_by_selection = selection.select(real_metrics, selector.n_trees)
    assert_trees_equal(np.array(real_trees)[indices_by_selection], trees_by_selector)


def assert_trees_equal(trees1, trees2):
    assert len(trees1) == len(trees2)
    for i in range(len(trees1)):
        assert_array_equal(trees1[i].feature, trees2[i].feature)


@pytest.mark.parametrize("selection", [Selection.Rank,
                                            Selection.Tournament,
                                            Selection.Roulette,
                                            Selection.StochasticUniform
                                            ])
def test_warning_that_there_are_less_than_n_trees(selector, real_metrics, real_trees, selection):
    selector.set_params(n_trees=4, selection=selection)
    with pytest.warns(UserWarning):
        selector.select(real_trees, real_metrics)


# ==============================================================================
# Selection
# ==============================================================================

def random_function():
    """Never can't be sure that function is not random"""
    return 9


def test_add_new_selection():
    Selection.add_new("MyNewName", random_function)
    assert Selection.MyNewName.select() == 9
