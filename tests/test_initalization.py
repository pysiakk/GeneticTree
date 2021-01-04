import os
os.chdir("../")

from tests.utils_testing import *


def test_initialize_full():
    initializer = Initializer(n_trees=20, initial_depth=5, initialization=Initialization.Full)
    trees = initializer.initialize(X=X, y=y, sample_weight=sample_weight, threshold=thresholds)
    assert isinstance(trees[0], Tree)
    assert len(trees) == 20
    assert max(trees[0].nodes_depth) == 5
    return trees


def test_initialize_full_depth_diff():
    initializer = Initializer(n_trees=20, initial_depth=8, initialization=Initialization.Full)
    trees = initializer.initialize(X=X, y=y, sample_weight=sample_weight, threshold=thresholds)
    assert isinstance(trees[0], Tree)
    assert len(trees) == 20
    assert max(trees[0].nodes_depth) == 8
    return trees


def test_initialize_half():
    initializer = Initializer(n_trees=20, initial_depth=5, initialization=Initialization.Half)
    trees = initializer.initialize(X=X, y=y, sample_weight=sample_weight, threshold=thresholds)
    assert isinstance(trees[0], Tree)
    assert len(trees) == 20
    assert 1 <= max(trees[0].nodes_depth) <= 5
    return trees


def test_initialize_split():
    initializer = Initializer(n_trees=20, initial_depth=5, initialization=Initialization.Split, split_prob=0.7)
    trees = initializer.initialize(X=X, y=y, sample_weight=sample_weight, threshold=thresholds)
    assert isinstance(trees[0], Tree)
    assert len(trees) == 20
    assert 1 <= max(trees[0].nodes_depth) <= 5
    return trees


if __name__ == '__main__':
    trees1 = test_initialize_full()
    trees2 = test_initialize_half()
    trees3 = test_initialize_split()

    print("\nFull initialization:")
    for tree in trees1:
        print(tree.nodes_depth)

    print("\nHalf-by-half initialization:")
    for tree in trees2:
        print(tree.nodes_depth)

    print("\nSplit initialization:")
    for tree in trees3:
        print(tree.nodes_depth)
