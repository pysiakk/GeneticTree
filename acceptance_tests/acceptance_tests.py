from tests.utils_testing import *

X = iris.data
y = iris.target


def test_train_then_predict():
    gt = GeneticTree(max_iter=10, mutation_prob=0.3)
    gt.fit(X, y)
    y_pred = gt.predict(X)
    assert y_pred.shape[0] == X.shape[0]
    unique_classes = set(np.unique(y))
    for i in range(y_pred.shape[0]):
        assert unique_classes.__contains__(y_pred[i])


def test_train_then_predict_proba():
    gt = GeneticTree(max_iter=10, initialization=Initialization.Full, initial_depth=10)
    gt.fit(X, y)
    y_pred = gt.predict_proba(X)
    assert y_pred.shape[0] == X.shape[0]
    unique_classes = set(np.unique(y))
    assert y_pred.shape[1] == len(unique_classes)
    for i in range(y_pred.shape[0]):
        assert_almost_equal(np.sum(y_pred[i]), 1)

