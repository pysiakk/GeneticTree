from tests.set_up_variables_and_imports import *
from tree.observations import Observations
from genetic_tree import GeneticTree


@pytest.fixture
def X_converted():
    return GeneticTree._check_X_(GeneticTree(), X, True)


def test_observations_creation(X_converted):
    obs = Observations(X_converted, y)


@pytest.fixture
def obs(X_converted):
    return Observations(X_converted, y)


def test_creating_leaves_array_simple(obs):
    obs.test_create_leaves_array_simple()


def test_creating_leaves_array_many(obs):
    obs.test_create_leaves_array_many()


def test_creating_leaves_array_complex(obs):
    obs.test_create_leaves_array_complex()


def test_empty_leaves_ids(obs):
    obs.test_empty_leaves_ids()
