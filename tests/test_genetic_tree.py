from tests.utils_testing import *

n_trees = 20
const_seed = np.random.randint(0, 10**8)

# ====================================================================================================
# Genetic Tree
# ====================================================================================================


def test_seed():
    """
    Assert that algorithm will create the same best trees with set seed
    """
    seed = np.random.randint(0, 10**8)
    gt = GeneticTree(seed=seed, n_trees=n_trees, max_iterations=3)
    gt.fit(X, y)
    tree: Tree = gt._best_tree_

    gt2 = GeneticTree(seed=seed, n_trees=n_trees, max_iterations=3)
    gt2.fit(X, y)
    tree2: Tree = gt2._best_tree_

    assert_array_equal(tree.feature, tree2.feature)
    assert_array_equal(tree.threshold, tree2.threshold)
    assert_array_equal(tree.nodes_depth, tree2.nodes_depth)
    assert_array_equal(tree.children_left, tree2.children_left)
    assert_array_equal(tree.children_right, tree2.children_right)
    assert_array_equal(tree.parent, tree2.parent)


def test_none_seed():
    """
    test if seed can be None -> and then np won't set seed
    """
    GeneticTree(seed=None)


def test_none_argument():
    """
    If one argument is none it should raise Error
    """
    with pytest.raises(ValueError):
        GeneticTree(n_trees=None)


@pytest.fixture
def genetic_tree():
    genetic_tree = GeneticTree(n_trees=10, max_iterations=3, remove_other_trees=False, remove_variables=False)
    return genetic_tree


@pytest.fixture
def genetic_tree_fitted(genetic_tree):
    genetic_tree.fit(X, y)
    return genetic_tree


@pytest.fixture
def X_converted(genetic_tree):
    return genetic_tree._check_X_(X, True)


def test_predict_exception_when_not_fit(genetic_tree):
    """
    When any tree is not fit it should not allow prediction
    """
    with pytest.raises(Exception):
        genetic_tree.predict(X)


def test_X_y_different_sizes(genetic_tree):
    """
    X and y should have the same number of observations
    """
    with pytest.raises(ValueError):
        genetic_tree.fit(X, np.concatenate([y, [1]]))


@pytest.mark.parametrize("function",
                         [GeneticTree.predict,
                          GeneticTree.predict_proba,
                          GeneticTree.apply,
                          GeneticTree._check_X_])
def test_X_with_less_features(genetic_tree, function):
    """
    X added as predicted should have the same number of features as
    X used in fit function
    """
    genetic_tree.fit(X, y)
    with pytest.raises(ValueError):
        function(genetic_tree, X[:, 1:3], False)


def test_set_n_features_after_check_input(genetic_tree):
    genetic_tree._check_input_(X, y, False)
    assert X.shape[1] == genetic_tree._n_features_


@pytest.fixture
def np_randint():
    np.random.seed(const_seed)
    return np.random.randint(0, 10**8)


def test_set_seed(np_randint):
    GeneticTree(seed=const_seed)
    assert np_randint == np.random.randint(0, 10**8)


def test_predict(genetic_tree_fitted, X_converted):
    assert_array_equal(genetic_tree_fitted.predict(X),
                       genetic_tree_fitted._best_tree_.predict(X_converted))


def test_predict_proba(genetic_tree_fitted, X_converted):
    assert_array_equal(genetic_tree_fitted.predict_proba(X).toarray(),
                       genetic_tree_fitted._best_tree_.predict_proba(X_converted).toarray())


def test_apply(genetic_tree_fitted, X_converted):
    assert_array_equal(genetic_tree_fitted.apply(X),
                       genetic_tree_fitted._best_tree_.apply(X_converted))


# +++++++++++++++
# Metric functions
# +++++++++++++++

def assert_last_metric(genetic_tree):
    assert genetic_tree.acc_best[-1] == np.max(Evaluator.get_accuracies(genetic_tree._trees_))
    assert genetic_tree.acc_mean[-1] == np.mean(Evaluator.get_accuracies(genetic_tree._trees_))
    assert genetic_tree.depth_best[-1] == np.min(Evaluator.get_depths(genetic_tree._trees_))
    assert genetic_tree.depth_mean[-1] == np.mean(Evaluator.get_depths(genetic_tree._trees_))
    assert genetic_tree.n_leaves_best[-1] == np.min(Evaluator.get_n_leaves(genetic_tree._trees_))
    assert genetic_tree.n_leaves_mean[-1] == np.mean(Evaluator.get_n_leaves(genetic_tree._trees_))


def test_append_metrics(X_converted):
    genetic_tree = GeneticTree(n_trees=10, max_iterations=0, remove_other_trees=False, remove_variables=False)
    genetic_tree.fit(X_converted, y)
    assert_last_metric(genetic_tree)


def test_append_metrics_more_iterations(X_converted):
    genetic_tree = GeneticTree(n_trees=10, max_iterations=1, remove_other_trees=False, remove_variables=False)
    for i in range(10):
        genetic_tree.fit(X_converted, y)
        assert_last_metric(genetic_tree)
