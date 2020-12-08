import os
os.chdir("../")

from genetic.initializer import Initializer, InitializationType
from tests.set_up_variables_and_imports import *
from tree.thresholds import prepare_thresholds_array
from genetic_tree import GeneticTree
from tree.tree import Tree

X = GeneticTree._check_X_(GeneticTree(), X, True)
thresholds = prepare_thresholds_array(10, X)


def test_initialize_random():
    initializer = Initializer(n_trees=20, initial_depth=5, initialization_type=InitializationType.Random)
    trees = initializer.initialize(X=X, y=y, threshold=thresholds)
    assert isinstance(trees[0], Tree)
    assert len(trees) == 20
    assert max(trees[0].nodes_depth) == 5
    return trees


def test_initialize_random_depth_diff():
    initializer = Initializer(n_trees=20, initial_depth=8, initialization_type=InitializationType.Random)
    trees = initializer.initialize(X=X, y=y, threshold=thresholds)
    assert isinstance(trees[0], Tree)
    assert len(trees) == 20
    assert max(trees[0].nodes_depth) == 8
    return trees


def test_initialize_half():
    initializer = Initializer(n_trees=20, initial_depth=5, initialization_type=InitializationType.Half)
    trees = initializer.initialize(X=X, y=y, threshold=thresholds)
    assert isinstance(trees[0], Tree)
    assert len(trees) == 20
    assert 1 <= max(trees[0].nodes_depth) <= 5
    return trees


def test_initialize_split():
    initializer = Initializer(n_trees=20, initial_depth=5, initialization_type=InitializationType.Split, split_prob=0.7)
    trees = initializer.initialize(X=X, y=y, threshold=thresholds)
    assert isinstance(trees[0], Tree)
    assert len(trees) == 20
    assert 1 <= max(trees[0].nodes_depth) <= 5
    return trees


if __name__ == '__main__':
    trees1 = test_initialize_random()
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
