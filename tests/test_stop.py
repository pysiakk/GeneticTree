from tests.utils_testing import *


def test_stop_max_false():
    stop = Stopper()
    assert not stop.stop([0.7])


def test_stop_max_true():
    stop = Stopper()
    stop.current_iteration = 1000000
    assert stop.stop([0.7])


def test_stop_imp_false():
    stop = Stopper(early_stopping=True, n_iter_no_change=5)
    stop.best_metric_hist = [0.1, 0.1, 0.1, 0.1, 0.2]
    assert not stop.stop([0.2])


def test_stop_imp_true():
    stop = Stopper(early_stopping=True, n_iter_no_change=5)
    stop.best_metric_hist = [0.2, 0.2, 0.2, 0.2, 0.2]
    assert stop.stop([0.2])
    assert stop.stop([0.1])


def test_stop_imp_not_full():
    stop = Stopper(early_stopping=True, n_iter_no_change=5)
    stop.best_metric_hist = [0.2, 0.2]
    assert not stop.stop([0.1])
