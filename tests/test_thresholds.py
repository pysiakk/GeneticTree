from .utils_testing import *


@pytest.fixture
def X_converted():
    return GeneticTree._check_X(GeneticTree(), X, True)


@pytest.fixture
def X_sparse(X_converted):
    return dok_matrix(X_converted).tocsr()


@pytest.mark.parametrize("n_thresholds", [2, 3, 5, 8, 13])
def test_thresholds_sparse_and_dense(X_sparse, X_converted, n_thresholds):
    thresholds_sparse = prepare_thresholds_array(n_thresholds, X_sparse)
    thresholds_dense = prepare_thresholds_array(n_thresholds, X_converted)
    assert_array_equal(thresholds_dense, thresholds_sparse)
