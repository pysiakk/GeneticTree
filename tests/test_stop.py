import os
os.chdir("../")

from genetic.initializer import Initializer, InitializationType
from tests.set_up_variables_and_imports import *
from tree.thresholds import prepare_thresholds_array
from genetic_tree import GeneticTree
from genetic.stop_condition import StopCondition
from tree.tree import Tree

X = GeneticTree._check_X_(GeneticTree(), X, True)
thresholds = prepare_thresholds_array(10, X)


def test_stop_max_false():
    stop = StopCondition()
    assert not stop.stop(0.7)


def test_stop_max_true():
    stop = StopCondition()
    stop.current_iteration = 1000000
    assert stop.stop(0.7)


def test_stop_imp_false():
    stop = StopCondition(use_without_improvement=True, max_iterations_without_improvement=5)
    stop.best_metric_hist = [0.1, 0.1, 0.1, 0.1, 0.2]
    assert not stop.stop(0.2)


def test_stop_imp_true():
    stop = StopCondition(use_without_improvement=True, max_iterations_without_improvement=5)
    stop.best_metric_hist = [0.2, 0.2, 0.2, 0.2, 0.2]
    assert stop.stop(0.2)
    assert stop.stop(0.1)


def test_stop_imp_not_full():
    stop = StopCondition(use_without_improvement=True, max_iterations_without_improvement=5)
    stop.best_metric_hist = [0.2, 0.2]
    assert not stop.stop(0.1)


if __name__ == '__main__':
    pass
