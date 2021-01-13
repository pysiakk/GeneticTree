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


def test_get_best_params():
    cross_prob = [0.2, 0.4, 0.6, 0.8]
    accuracy_best = []
    for i in range(len(cross_prob)):
        gt = GeneticTree(max_iter=10, cross_prob=cross_prob[i])
        gt.fit(X, y)
        accuracy_best.append(gt.acc_best[-1])
    assert len(accuracy_best) == len(cross_prob)
    best_accuracy_id = np.argmax(np.array(accuracy_best))
    print(f"Best accuracy is for cross prob: {cross_prob[best_accuracy_id]}")


def test_train_model_many_times():
    gt = GeneticTree(max_iter=10, keep_last_population=True)
    gt.fit(X, y)
    for i in range(10):
        weights = np.ones(150)
        print(f"Score: {np.sum(gt.predict(X) == y)}")
        weights[gt.predict(X) != y] += 0.5
        gt.partial_fit(X, y, sample_weight=weights)
