from genetic_tree import GeneticTree
from tests.set_up_variables_and_imports import *

# overwrite dataset to not be in random order
iris = datasets.load_iris()


def test_performance():
    X = iris.data
    y = iris.target
    n_trees = 20
    gt = GeneticTree(n_trees=n_trees, max_trees=300, max_iterations=30, remove_other_trees=False, remove_variables=False)
    gt.fit(X, y)
    X = gt.genetic_processor.forest.X

    tree: Tree = gt.genetic_processor.forest.trees[5]
    print(f'Score: {gt.genetic_processor.selector.__evaluate_single_tree__(tree, X)/150}')
    for k, val in tree.observations.items():
        print(f'\n Node id: {k}')
        for v in val:
            print(f'Current class: {v.current_class}, proper class: {v.proper_class}, observation id: {v.observation_id}')

    score_sum = 0
    score_sum_by_prediction = 0
    for i in range(n_trees):
        score_sum += gt.genetic_processor.selector.__evaluate_single_tree__(gt.genetic_processor.forest.trees[i], X)
        y_pred = gt.genetic_processor.forest.trees[i].predict(X)
        score_sum_by_prediction += np.sum(y_pred == y)

    print(f'Mean score: {score_sum / 150 / n_trees}')
    print(f'Best score: {np.sum(gt.predict(X) == y) / 150}')

    # test if prediction works
    assert score_sum == score_sum_by_prediction

    # test if high level predict returns the same array as tree.predict()
    forest: Forest = gt.genetic_processor.forest
    assert_array_equal(gt.predict(X), forest.best_tree.predict(X))


if __name__ == "__main__":
    test_performance()
