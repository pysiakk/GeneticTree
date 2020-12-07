# general imports
import copy
import io
import numpy as np
import pickle
import psutil
import pytest
from threading import Thread
import time

# sklearn imports (mainly assertions)
from sklearn import datasets
from sklearn.utils._testing import assert_allclose
from sklearn.utils._testing import assert_array_equal
from sklearn.utils._testing import assert_array_almost_equal
from sklearn.utils._testing import assert_almost_equal
from sklearn.utils._testing import assert_warns
from sklearn.utils._testing import assert_warns_message
from sklearn.utils._testing import create_memmap_backed_data
from sklearn.utils._testing import ignore_warnings

# low level (Cython) imports
from tree._utils import test_copy_int_array, test_copy_leaves
from tree.tree import Tree, copy_tree, test_independence_of_copied_tree
from tree.thresholds import prepare_thresholds_array
from tree.observations import Observations, copy_observations
from tree.builder import Builder, FullTreeBuilder, test_add_node, test_add_leaf
from tree.mutator import mutate_random_node, mutate_random_class_or_threshold
from tree.mutator import mutate_random_feature, mutate_random_threshold
from tree.mutator import mutate_random_class
from tree.crosser import cross_trees
from tree.evaluation import get_accuracies, get_trees_depths, get_trees_n_leaves


# high level (Python) imports
from genetic.initializer import Initializer, InitializationType
from genetic.mutator import Mutator, MutationType
from genetic.evaluator import Evaluator, Metric
from genetic.evaluator import get_accuracy, get_accuracy_and_n_leaves, get_accuracy_and_depth
from genetic.selector import SelectionType, Selector
from genetic.selector import get_selected_indices_by_rank_selection
from genetic.selector import get_selected_indices_by_tournament_selection
from genetic.selector import get_selected_indices_by_roulette_selection
from genetic.selector import get_selected_indices_by_stochastic_uniform_selection

# package interface
from genetic_tree import GeneticTree

# import constants
from tree.tree import TREE_UNDEFINED

# load iris dataset and randomly permute it (example from sklearn.tree.test.test_tree)
iris = datasets.load_iris()
rng = np.random.RandomState(1)
perm = rng.permutation(iris.target.size)

X = iris.data[perm]
y = iris.target[perm]
y = np.ascontiguousarray(y, dtype=np.intp)

n_thresholds: int = 2
X = GeneticTree._check_X_(GeneticTree(), X, True)

# thresholds array have unique values
# it is needed to proper test mutating thresholds
thresholds = prepare_thresholds_array(n_thresholds, X)


def initialize_iris_tree():
    return Tree(np.unique(y).shape[0], X, y, thresholds, np.random.randint(10 ** 8))


def build_trees(depth: int = 1, n_trees: int = 10):
    builder: FullTreeBuilder = FullTreeBuilder()
    trees = []
    for i in range(n_trees):
        tree: Tree = initialize_iris_tree()
        tree.resize_by_initial_depth(depth)
        builder.build(tree, depth)
        tree.initialize_observations()
        trees.append(tree)
    return trees
