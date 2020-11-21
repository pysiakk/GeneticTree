from tests.set_up_variables_and_imports import *
from tree.observations import Observations
from genetic_tree import GeneticTree
from tree.thresholds import prepare_thresholds_array
from tree.builder import FullTreeBuilder


@pytest.fixture
def X_converted():
    return GeneticTree._check_X_(GeneticTree(), X, True)


def test_observations_creation(X_converted):
    obs = Observations(X_converted, y)


@pytest.fixture
def obs(X_converted):
    return Observations(X_converted, y)


@pytest.fixture
def tree(X_converted):
    thresholds = prepare_thresholds_array(10, X_converted)
    tree = Tree(3, X_converted, y, thresholds)
    FullTreeBuilder.build(FullTreeBuilder(), tree, 10)
    return tree


# ==============================================================================
# Inner structures
# ==============================================================================

def test_creating_leaves_array_simple(obs):
    obs.test_create_leaves_array_simple()


def test_creating_leaves_array_many(obs):
    obs.test_create_leaves_array_many()


def test_creating_leaves_array_complex(obs):
    obs.test_create_leaves_array_complex()


def test_empty_leaves_ids(obs):
    obs.test_empty_leaves_ids()


def test_copy_to_leaves_to_reassign(obs):
    obs.test_copy_to_leaves_to_reassign()


def test_delete_leaves_to_reassign(obs):
    obs.test_delete_leaves_to_reassign()


# ==============================================================================
# Observations
# ==============================================================================

def test_initialization(obs, tree):
    obs.test_initialization(tree)

