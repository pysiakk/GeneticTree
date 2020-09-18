from genetic_tree import GeneticTree
from tests.set_up_variables_and_imports import *

# overwrite dataset to not be in random order
iris = datasets.load_iris()
X = iris.data
y = iris.target

n_trees = 200
gt = GeneticTree(n_trees=n_trees, max_trees=300, max_iterations=300)
gt.fit(X, y)
X = gt.genetic_processor.forest.X

tree: Tree = gt.genetic_processor.forest.trees[5]
print(f'Score: {gt.genetic_processor.selector.__evaluate_single_tree__(tree, X)/150}')
for k, val in tree.observations.items():
    print(f'\n Node id: {k}')
    for v in val:
        print(f'Current class: {v.current_class}, proper class: {v.proper_class}, observation id: {v.observation_id}')

score_sum = 0
for i in range(n_trees):
    score_sum += gt.genetic_processor.selector.__evaluate_single_tree__(gt.genetic_processor.forest.trees[i], X)

print(f'Mean score: {score_sum / 150 / n_trees}')