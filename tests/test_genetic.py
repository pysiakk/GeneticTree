from genetic_tree import GeneticTree
from tests.set_up_variables_and_imports import *


def initialize_tree() -> Tree:
    tree: Tree = Tree(5, 3, 3)
    builder: FullTreeBuilder = FullTreeBuilder(3)
    builder.build(tree, X, y)
    return tree


def initialize_forest(max_depth: int = 3) -> Forest:
    gt = GeneticTree(max_depth=max_depth)
    gt.fit(X, y)
    return gt.genetic_processor.forest


def initialization():
    forest: Forest = initialize_forest(7)
    tree: Tree = initialize_tree()
    tree_example: Tree = forest.trees[0]
    print(f'Features of tree: {tree.feature}\n')
    print(f'Features of first forest tree: {tree_example.feature},\n'
          f'First 100 features: {tree_example.feature[:100]},\n'
          f'And number of features {tree_example.feature.shape}')


def mutate_feature():
    gt = GeneticTree(change_feature=1, max_depth=1, max_iterations=1)
    gt.fit(X, y)
    forest_before = gt.genetic_processor.forest
    gt.genetic_processor.crosser.cross_population(forest_before)
    print(f'Features of tree before mutation: {forest_before.trees[0].feature}')
    forest_after = gt.genetic_processor.mutator.mutate(forest_before)
    print(f'Features of tree in previous forest: {forest_before.trees[0].feature}')
    print(f'Features of tree after mutation: {forest_after.trees[0].feature}')
    return forest_before


if __name__ == "__main__":
    forest: Forest = mutate_feature()
    for tree in forest.trees[1000:1040]:
        print(tree.feature)
