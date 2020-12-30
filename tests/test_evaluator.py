from tests.utils_testing import *

# ==============================================================================
# Metric functions
# ==============================================================================


@pytest.fixture
def trees():
    trees = []
    for i in range(20):
        trees.append(build_trees(5, 1)[0])
    return trees


def test_get_accuracy(trees):
    metrics = np.empty(20, float)
    for i in range(20):
        metrics[i] = trees[i].proper_classified / trees[i].n_observations
    assert_array_equal(metrics, get_accuracy(trees))


@pytest.mark.parametrize("n_leaves_factor", [0.01, 0.001, 0.0001])
def test_get_accuracy_and_n_leaves(trees, n_leaves_factor):
    metrics = np.empty(20, float)
    for i in range(20):
        metrics[i] = trees[i].proper_classified / trees[i].n_observations \
                     - n_leaves_factor * (trees[i].node_count + 1) / 2
    assert_array_equal(metrics, get_accuracy_and_n_leaves(trees, n_leaves_factor))


@pytest.mark.parametrize("depth_factor", [0.01, 0.001, 0.0001])
def test_get_accuracy_and_depth(trees, depth_factor):
    metrics = np.empty(20, float)
    for i in range(20):
        metrics[i] = trees[i].proper_classified / trees[i].n_observations \
                     - depth_factor * trees[i].depth
    assert_array_equal(metrics, get_accuracy_and_depth(trees, depth_factor))


# ==============================================================================
# Evaluator
# ==============================================================================

# Base evaluator
@pytest.fixture
def evaluator():
    return Evaluator(Metric.AccuracyMinusDepth)


# +++++++++++++++
# Init and set params
# +++++++++++++++

@pytest.mark.parametrize("metric", [Metric.Accuracy, Metric.AccuracyMinusDepth, Metric.AccuracyMinusLeavesNumber])
def test_evaluator_init(metric):
    evaluator = Evaluator(metric)
    assert evaluator.metric == metric


@pytest.mark.parametrize("metric", ["string", [], 4])
def test_evaluator_init_metric_wrong_type(metric):
    with pytest.raises(TypeError):
        evaluator = Evaluator(metric)


@pytest.mark.parametrize("metric", [Metric.Accuracy, Metric.AccuracyMinusDepth, Metric.AccuracyMinusLeavesNumber])
def test_set_metric(evaluator, metric):
    evaluator.set_params(metric=metric)
    assert evaluator.metric == metric


@pytest.mark.parametrize("metric", ["string", [], 4])
def test_set_metric_wrong_type(evaluator, metric):
    with pytest.raises(TypeError):
        evaluator.set_params(metric=metric)


def test_set_new_metric(evaluator):
    Metric.add_new("NewMetric", lambda x: x)
    evaluator.set_params(metric=Metric.NewMetric)
    assert str(type(evaluator.metric)) == str(Metric)


def test_set_params_kwargs(evaluator):
    evaluator.set_params(a=2)
    assert evaluator._kwargs['a'] == 2
    evaluator.set_params(a=3)
    assert evaluator._kwargs['a'] == 3

# +++++++++++++++
# Evaluate
# +++++++++++++++

@pytest.mark.parametrize("metric", [Metric.Accuracy, Metric.AccuracyMinusDepth, Metric.AccuracyMinusLeavesNumber])
def test_evaluate(evaluator, metric, trees):
    evaluator.set_params(metric=metric)
    assert_array_equal(evaluator.evaluate(trees), metric.evaluate(trees))


# +++++++++++++++
# Get best tree
# +++++++++++++++

@pytest.mark.parametrize("metric", [Metric.Accuracy, Metric.AccuracyMinusDepth, Metric.AccuracyMinusLeavesNumber])
def test_get_best_tree(evaluator, metric, trees):
    evaluator.set_params(metric=metric)
    metrics = evaluator.evaluate(trees)
    best_index = np.argmax(metrics)
    assert evaluator.get_best_tree_index(trees) == best_index


# +++++++++++++++
# Metric functions
# +++++++++++++++

def test_get_accuracies(evaluator, trees):
    assert_array_equal(evaluator.get_accuracies(trees), get_accuracies(trees))


def test_get_depths(evaluator, trees):
    assert_array_equal(evaluator.get_depths(trees), get_trees_depths(trees))


def test_get_n_leaves(evaluator, trees):
    assert_array_equal(evaluator.get_n_leaves(trees), get_trees_n_leaves(trees))
