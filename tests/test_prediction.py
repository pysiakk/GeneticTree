from tests.utils_testing import *

n_trees = 20


@pytest.fixture
def genetic_tree():
    gt = GeneticTree(n_trees=n_trees, max_iter=30, keep_last_population=True, remove_variables=False)
    gt.fit(iris.data, iris.target)
    return gt


@pytest.fixture
def X(genetic_tree):
    return genetic_tree._trees[0].X


def print_observation(genetic_tree):
    # TODO change into tests
    # for example test that node_id==-1 should have empty list
    tree: Tree = genetic_tree._trees[5]
    print(f'Score: {tree.proper_classified / 150}')
    for k, val in tree.observations.items():
        print(f'\n Node id: {k}')
        for v in val:
            print(f'Current class: {v.current_class}, proper class: {v.proper_class}, observation id: {v.observation_id}')


def test_score(genetic_tree, X):
    score_sum = 0
    score_sum_by_prediction = 0
    for i in range(n_trees):
        tree: Tree = genetic_tree._trees[i]
        score_sum += tree.proper_classified
        y_pred = genetic_tree._trees[i].test_predict(X)
        score_sum_by_prediction += np.sum(y_pred == iris.target)
    # test if prediction works
    assert score_sum == score_sum_by_prediction


def test_high_level_and_low_level_prediction(genetic_tree, X):
    # test if high level predict returns the same array as tree.predict()
    assert_array_equal(genetic_tree.predict(X), genetic_tree._best_tree.test_predict(X))


if __name__ == "__main__":
    gt = GeneticTree(keep_last_population=True, remove_variables=False, max_iter=30)
    gt.fit(iris.data, iris.target)
    tree = gt._best_tree
    print(tree.proper_classified)
