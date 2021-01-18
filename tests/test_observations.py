from tests.utils_testing import *


@pytest.fixture
def X_converted():
    return GeneticTree._check_X(GeneticTree(), X, True)


def test_observations_creation(X_converted):
    obs = Observations(X_converted, y, sample_weight)


@pytest.fixture
def obs(X_converted):
    return Observations(X_converted, y, sample_weight)


@pytest.fixture
def tree(X_converted):
    tree = initialize_iris_tree()
    full_tree_builder(tree, 10)
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


def test_removing_and_reassigning(obs, tree):
    obs.test_removing_and_reassigning(tree)


def test_pickling(obs, tree):
    obs.test_initialization(tree)
    bytes_io = io.BytesIO()
    pickle.dump(obs, bytes_io)
    bytes_io.seek(0)
    observations = pickle.load(bytes_io)
    assert observations.proper_classified == obs.proper_classified
    assert observations.n_observations == obs.n_observations
    # TODO: assertions about leaves, empty_leaves_ids


def test_copy_observations(obs, tree):
    obs.test_removing_and_reassigning(tree)
    obs_new = copy_observations(obs)
    assert obs.proper_classified == obs_new.proper_classified
    assert obs.n_observations == obs_new.n_observations


# ==============================================================================
# Utils
# ==============================================================================

def test_copy_int_array_():
    test_copy_int_array()


def test_copy_leaves_():
    test_copy_leaves()


# ==============================================================================
# LeafFinder
# ==============================================================================

@pytest.fixture
def X_sparse(X_converted):
    return dok_matrix(X_converted).tocsr()


def test_find_leaf_dense(X_converted, tree):
    leaf_finder = LeafFinder(X_converted)
    leaf_finder.test_find_leaf_dense(tree)


def test_find_leaf_sparse(X_sparse, tree):
    leaf_finder = LeafFinder(X_sparse)
    leaf_finder.test_find_leaf_sparse(tree)


def test_find_leaf_sparse_and_dense_equal(X_converted, X_sparse, tree):
    leaf_finder = LeafFinder(X_converted)
    y_dense = leaf_finder.test_find_leaves(tree)
    leaf_finder = LeafFinder(X_sparse)
    y_sparse = leaf_finder.test_find_leaves(tree)
    assert_array_equal(y_dense, y_sparse)
