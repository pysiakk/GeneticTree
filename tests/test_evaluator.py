from genetic.evaluator import Evaluator, Metric
from genetic.evaluator import get_accuracy, get_accuracy_and_size, get_accuracy_and_depth

from tests.test_tree import build
from sklearn.utils._testing import assert_array_equal

import numpy as np
import pytest


# ==============================================================================
# Metric functions
# ==============================================================================

@pytest.fixture
def trees():
    trees = []
    for i in range(20):
        trees.append(build(5, 1)[0][0])
    return trees


def test_get_accuracy(trees):
    metrics = np.empty(20, float)
    for i in range(20):
        metrics[i] = trees[i].get_proper_classified() / trees[i].n_observations
    assert_array_equal(metrics, get_accuracy(trees))


@pytest.mark.parametrize("size_factor", [0.01, 0.001, 0.0001])
def test_get_accuracy_and_size(trees, size_factor):
    metrics = np.empty(20, float)
    for i in range(20):
        metrics[i] = trees[i].get_proper_classified() / trees[i].n_observations \
                     - size_factor * trees[i].node_count
    assert_array_equal(metrics, get_accuracy_and_size(trees, size_factor))


@pytest.mark.parametrize("depth_factor", [0.01, 0.001, 0.0001])
def test_get_accuracy_and_size(trees, depth_factor):
    metrics = np.empty(20, float)
    for i in range(20):
        metrics[i] = trees[i].get_proper_classified() / trees[i].n_observations \
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

@pytest.mark.parametrize("metric", [Metric.Accuracy, Metric.AccuracyMinusDepth, Metric.AccuracyMinusSize])
def test_evaluator_init(metric):
    evaluator = Evaluator(metric)
    assert evaluator.metric == metric


@pytest.mark.parametrize("metric", ["string", [], 4])
def test_evaluator_init_metric_wrong_type(metric):
    with pytest.raises(TypeError):
        evaluator = Evaluator(metric)


@pytest.mark.parametrize("metric", [Metric.Accuracy, Metric.AccuracyMinusDepth, Metric.AccuracyMinusSize])
def test_set_metric(evaluator, metric):
    evaluator.set_params(metric=metric)
    assert evaluator.metric == metric


@pytest.mark.parametrize("metric", ["string", [], 4])
def test_set_metric_wrong_type(evaluator, metric):
    with pytest.raises(TypeError):
        evaluator.set_params(metric=metric)


# +++++++++++++++
# Evaluate
# +++++++++++++++

@pytest.mark.parametrize("metric", [Metric.Accuracy, Metric.AccuracyMinusDepth, Metric.AccuracyMinusSize])
def test_evaluate(evaluator, metric, trees):
    evaluator.set_params(metric=metric)
    assert_array_equal(evaluator.evaluate(trees), metric.evaluate(trees))


# +++++++++++++++
# Get best tree
# +++++++++++++++

@pytest.mark.parametrize("metric", [Metric.Accuracy, Metric.AccuracyMinusDepth, Metric.AccuracyMinusSize])
def test_get_best_tree(evaluator, metric, trees):
    evaluator.set_params(metric=metric)
    metrics = evaluator.evaluate(trees)
    best_index = np.argmax(metrics)
    assert evaluator.get_best_tree_index(trees) == best_index