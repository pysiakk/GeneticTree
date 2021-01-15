# general imports
import copy
import io
import numpy as np
from scipy.sparse import dok_matrix
import pickle
import psutil
import pytest
from threading import Thread
import time
import math

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
from genetic_tree.tree._utils import test_copy_int_array, test_copy_leaves
from genetic_tree.tree.tree import Tree, copy_tree, test_independence_of_copied_tree
from genetic_tree.tree.thresholds import prepare_thresholds_array
from genetic_tree.tree.observations import Observations, copy_observations, LeafFinder
from genetic_tree.tree.builder import full_tree_builder, split_tree_builder, test_add_node, test_add_leaf
from genetic_tree.tree.mutator import mutate_random_node, mutate_random_class_or_threshold
from genetic_tree.tree.mutator import mutate_random_feature, mutate_random_threshold
from genetic_tree.tree.mutator import mutate_random_class
from genetic_tree.tree.mutator import test_mutate_feature, test_mutate_class, test_mutate_threshold
from genetic_tree.tree.crosser import cross_trees
from genetic_tree.tree.evaluation import get_accuracies, get_trees_depths, get_trees_n_leaves


# high level (Python) imports
from genetic_tree import Initializer, Initialization
from genetic_tree import Mutator, Mutation
from genetic_tree import Crosser
from genetic_tree import Evaluator, Metric
from genetic_tree.genetic.evaluator import get_accuracy, get_accuracy_and_n_leaves, get_accuracy_and_depth
from genetic_tree import Selection, Selector
from genetic_tree.genetic.selector import metrics_greater_than_zero
from genetic_tree.genetic.selector import get_selected_indices_by_rank_selection
from genetic_tree.genetic.selector import get_selected_indices_by_tournament_selection
from genetic_tree.genetic.selector import get_selected_indices_by_roulette_selection
from genetic_tree.genetic.selector import get_selected_indices_by_stochastic_uniform_selection
from genetic_tree import Stopper

# package interface
from genetic_tree import GeneticTree

# import constants
from genetic_tree.tree.tree import TREE_UNDEFINED

# load iris dataset and randomly permute it (example from sklearn.tree.test.test_tree)
iris = datasets.load_iris()
rng = np.random.RandomState(1)
perm = rng.permutation(iris.target.size)

X = iris.data[perm]
y = iris.target[perm]
y = np.ascontiguousarray(y, dtype=np.intp)

n_thresholds: int = 2
X = GeneticTree._check_X(GeneticTree(), X, True)

# thresholds array have unique values
# it is needed to proper test mutating thresholds
thresholds = prepare_thresholds_array(n_thresholds, X)

sample_weight = np.ascontiguousarray(np.ones(150), dtype=np.float32)


def initialize_iris_tree():
    return Tree(np.unique(y), X, y, sample_weight, thresholds, np.random.randint(10 ** 8))


def build_trees(depth: int = 1, n_trees: int = 10):
    trees = []
    for i in range(n_trees):
        tree: Tree = initialize_iris_tree()
        tree.resize_by_initial_depth(depth)
        full_tree_builder(tree, depth)
        tree.initialize_observations()
        trees.append(tree)
    return trees
