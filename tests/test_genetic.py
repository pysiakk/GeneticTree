import os
os.chdir("../")

from genetic_tree import GeneticTree
from tests.set_up_variables_and_imports import *
from tree.thresholds import prepare_thresholds_array

X = GeneticTree._check_X_(GeneticTree(), X, True)
thresholds = prepare_thresholds_array(10, X)


def initialize_tree(thresholds) -> Tree:
    tree: Tree = Tree(3, X, y, thresholds)
    tree.resize_by_initial_depth(3)
    builder: FullTreeBuilder = FullTreeBuilder(3)
    builder.build(tree)
    tree.initialize_observations(X, y)
    return tree


def initialize_forest(initial_depth: int = 3) -> Forest:
    gt = GeneticTree(initial_depth=initial_depth, remove_variables=False, remove_other_trees=False, max_iterations=1)
    gt.fit(X, y)
    return gt.forest


def initialization():
    forest: Forest = initialize_forest(7)

    tree: Tree = initialize_tree(thresholds)
    tree_example: Tree = forest.trees[0]
    print(f'Features of tree: {tree.feature}\n')
    print(f'Features of first forest tree: {tree_example.feature},\n'
          f'First 100 features: {tree_example.feature[:100]},\n'
          f'And number of features {tree_example.feature.shape}')


def mutate_feature():
    gt = GeneticTree(feature_prob=1, initial_depth=1, max_iterations=1,
                     remove_other_trees=False, remove_variables=False)
    gt.fit(X, y)
    forest_before = gt.forest
    gt.crosser.cross_population(forest_before)
    print(f'Features of tree before mutation: {forest_before.trees[0].feature}')
    forest_after = gt.mutator.mutate(forest_before)
    print(f'Features of tree in previous forest: {forest_before.trees[0].feature}')
    print(f'Features of tree after mutation: {forest_after.trees[0].feature}')
    return forest_before


if __name__ == "__main__":
    forest: Forest = mutate_feature()
    for tree in forest.trees[1000:1040]:
        print(tree.feature)
