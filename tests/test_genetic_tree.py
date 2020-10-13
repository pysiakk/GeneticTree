from genetic_tree import GeneticTree
from tests.set_up_variables_and_imports import *

import os
os.chdir("../")

n_trees = 20


def test_seed():
    """
    Assert that algorithm will create the same best trees with set seed
    """
    seed = np.random.randint(0, 10**8)
    gt = GeneticTree(seed=seed, n_trees=n_trees, max_iterations=3)
    gt.fit(X, y)
    tree: Tree = gt.genetic_processor.forest.best_tree

    gt2 = GeneticTree(seed=seed, n_trees=n_trees, max_iterations=3)
    gt2.fit(X, y)
    tree2: Tree = gt2.genetic_processor.forest.best_tree

    assert_array_equal(tree.feature, tree2.feature)
    assert_array_equal(tree.threshold, tree2.threshold)
    assert_array_equal(tree.nodes_depth, tree2.nodes_depth)
    assert_array_equal(tree.children_left, tree2.children_left)
    assert_array_equal(tree.children_right, tree2.children_right)
    assert_array_equal(tree.parent, tree2.parent)

