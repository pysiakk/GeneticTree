from tests.utils_testing import *

n_trees = 20
const_seed = np.random.randint(0, 10**8)


# ====================================================================================================
# Genetic Tree
# ====================================================================================================

def assert_trees_equal(tree, tree2):
    assert_array_equal(tree.feature, tree2.feature)
    assert_array_equal(tree.threshold, tree2.threshold)
    assert_array_equal(tree.nodes_depth, tree2.nodes_depth)
    assert_array_equal(tree.children_left, tree2.children_left)
    assert_array_equal(tree.children_right, tree2.children_right)
    assert_array_equal(tree.parent, tree2.parent)


def test_seed():
    """
    Assert that algorithm will create the same best trees with set seed
    """
    seed = np.random.randint(0, 10**8)
    gt = GeneticTree(random_state=seed, n_trees=n_trees, max_iter=3)
    gt.fit(X, y)
    tree: Tree = gt._best_tree

    gt2 = GeneticTree(random_state=seed, n_trees=n_trees, max_iter=3)
    gt2.fit(X, y)
    tree2: Tree = gt2._best_tree

    assert_trees_equal(tree, tree2)


def test_none_seed():
    """
    test if seed can be None -> and then np won't set seed
    """
    GeneticTree(random_state=None)


def test_none_argument():
    """
    If one argument is none it should raise Error
    """
    with pytest.raises(ValueError):
        GeneticTree(n_trees=None)


def test_verbose(genetic_tree):
    genetic_tree.set_params(verbose=1).fit(X, y)


@pytest.fixture
def genetic_tree() -> GeneticTree:
    genetic_tree = GeneticTree(n_trees=10, max_iter=3, keep_last_population=True, remove_variables=False)
    return genetic_tree


@pytest.fixture
def genetic_tree_fitted(genetic_tree):
    genetic_tree.fit(X, y)
    return genetic_tree


@pytest.fixture
def X_converted(genetic_tree):
    return genetic_tree._check_X(X, True)


@pytest.fixture
def X_sparse(X_converted):
    return dok_matrix(X_converted).tocsr()


def test_fit_dense_and_sparse(X_sparse):
    random_state = np.random.randint(0, 10**8)
    best_dense: Tree = GeneticTree(max_iter=1, n_trees=10, random_state=random_state).fit(X, y)._best_tree
    best_sparse: Tree = GeneticTree(max_iter=1, n_trees=10, random_state=random_state).fit(X_sparse, y)._best_tree
    assert_trees_equal(best_dense, best_sparse)


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
                          GeneticTree._check_X])
def test_X_with_less_features(genetic_tree, function):
    """
    X added as predicted should have the same number of features as
    X used in fit function
    """
    genetic_tree.fit(X, y)
    with pytest.raises(ValueError):
        function(genetic_tree, X[:, 1:3], False)


def test_set_n_features_after_check_input(genetic_tree):
    genetic_tree._check_input(X, y, None, check_input=False)
    assert X.shape[1] == genetic_tree._n_features


@pytest.fixture
def np_randint():
    np.random.seed(const_seed)
    return np.random.randint(0, 10**8)


def test_set_seed(np_randint):
    GeneticTree(random_state=const_seed)
    assert np_randint == np.random.randint(0, 10**8)


def test_predict(genetic_tree_fitted, X_converted):
    assert_array_equal(genetic_tree_fitted.predict(X),
                       genetic_tree_fitted._best_tree.test_predict(X_converted))


def test_predict_proba(genetic_tree_fitted, X_converted):
    assert_array_equal(genetic_tree_fitted.predict_proba(X),
                       genetic_tree_fitted._best_tree.test_predict_proba(X_converted))


def test_apply(genetic_tree_fitted, X_converted):
    assert_array_equal(genetic_tree_fitted.apply(X),
                       genetic_tree_fitted._best_tree.apply(X_converted))


def test_apply_sparse_and_dense(genetic_tree_fitted, X_sparse):
    assert_array_equal(genetic_tree_fitted.apply(X),
                       genetic_tree_fitted.apply(X_sparse))

# +++++++++++++++
# Metric functions
# +++++++++++++++

def assert_last_metric(genetic_tree):
    best_tree_index = genetic_tree.evaluator.get_best_tree_index(genetic_tree._trees)
    assert genetic_tree.acc_best[-1] == Evaluator.get_accuracies(genetic_tree._trees)[best_tree_index]
    assert genetic_tree.acc_mean[-1] == np.mean(Evaluator.get_accuracies(genetic_tree._trees))
    assert genetic_tree.depth_best[-1] == Evaluator.get_depths(genetic_tree._trees)[best_tree_index]
    assert genetic_tree.depth_mean[-1] == np.mean(Evaluator.get_depths(genetic_tree._trees))
    assert genetic_tree.n_leaves_best[-1] == Evaluator.get_n_leaves(genetic_tree._trees)[best_tree_index]
    assert genetic_tree.n_leaves_mean[-1] == np.mean(Evaluator.get_n_leaves(genetic_tree._trees))
    assert genetic_tree.metric_best[-1] == genetic_tree.evaluator.evaluate(genetic_tree._trees)[best_tree_index]
    assert genetic_tree.metric_mean[-1] == np.mean(genetic_tree.evaluator.evaluate(genetic_tree._trees))


def test_append_metrics(X_converted):
    genetic_tree = GeneticTree(n_trees=10, max_iter=0, keep_last_population=True, remove_variables=False)
    genetic_tree.fit(X_converted, y)
    assert_last_metric(genetic_tree)


def test_append_metrics_more_iterations(X_converted):
    genetic_tree = GeneticTree(n_trees=10, max_iter=1, keep_last_population=True, remove_variables=False)
    genetic_tree.fit(X_converted, y)
    assert_last_metric(genetic_tree)
    for i in range(10):
        genetic_tree.partial_fit(X_converted, y)
        assert_last_metric(genetic_tree)
        assert len(genetic_tree.acc_best) == i*2+4


# +++++++++++++++
# Check input
# +++++++++++++++

def test_ones_as_sample_weight(genetic_tree, X_converted):
    *_, sample_weight = genetic_tree._check_input(X_converted, y, None, True)
    assert_array_equal(sample_weight, np.ones(150))


def test_converting_sample_weight(genetic_tree, X_converted):
    sample_weight = np.random.random(150)
    *_, sample_weight = genetic_tree._check_input(X_converted, y, sample_weight, True)
    assert sample_weight.shape[0] == 150
    assert sample_weight.dtype == np.float32
    assert sample_weight.flags.contiguous


def test_sample_weight_exception(genetic_tree, X_converted):
    sample_weight = np.random.random(149)
    with pytest.raises(ValueError):
        genetic_tree._check_input(X_converted, y, sample_weight, True)


def test_check_classes(genetic_tree, X_converted):
    assert genetic_tree._classes is None
    genetic_tree._check_input(X_converted, y, None, True)
    assert_array_equal(genetic_tree._classes, np.array([0, 1, 2]))
    y_copy = copy.copy(y)
    y_copy[y_copy == 1] = 3
    with pytest.raises(ValueError):
        genetic_tree._check_input(X_converted, y_copy, None, True)


# +++++++++++++++
# Classes
# +++++++++++++++

@pytest.mark.parametrize("X, y", [(X, y)])
def test_fit_tree_with_classes_not_starting_from_0(X, y):
    y_copied = copy.deepcopy(y)
    y_copied[y_copied == 2] = 8
    y_copied[y_copied == 1] = -3
    gt = GeneticTree(n_trees=20, max_iter=3)
    gt.fit(X, y_copied)
    assert set(np.unique(gt.predict(X))).issubset(np.array([-3, 0, 8]))


# +++++++++++++++
# Fit with using previous trees
# +++++++++++++++

def test_usage_of_previous_trees(genetic_tree, X_converted):
    genetic_tree.fit(X_converted, y)
    assert genetic_tree._trees is not None
    genetic_tree.fit(X_converted[:10, :], y[:10])
    assert genetic_tree._best_tree.X.shape[0] == 10
    assert genetic_tree._trees[0].X.shape[0] == 10


def test_usage_of_previous_best_tree(genetic_tree, X_converted):
    genetic_tree.set_params(keep_last_population=False)
    genetic_tree.fit(X_converted, y)
    assert genetic_tree._trees is None
    genetic_tree.set_params(keep_last_population=True)
    genetic_tree.fit(X_converted[:10, :], y[:10])
    assert genetic_tree._best_tree.X.shape[0] == 10
    assert genetic_tree._trees[0].X.shape[0] == 10

