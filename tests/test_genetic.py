from genetic_tree import GeneticTree
from tree.tree import Tree
from tree.forest import Forest
from tree.builder import FullTreeBuilder

from sklearn import datasets
import numpy as np

from sklearn.utils._testing import assert_allclose
from sklearn.utils._testing import assert_array_equal
from sklearn.utils._testing import assert_array_almost_equal
from sklearn.utils._testing import assert_almost_equal
from sklearn.utils._testing import assert_warns
from sklearn.utils._testing import assert_warns_message
from sklearn.utils._testing import create_memmap_backed_data
from sklearn.utils._testing import ignore_warnings

# load iris dataset and randomly permute it (example from sklearn.tree.test.test_tree)
iris = datasets.load_iris()
rng = np.random.RandomState(1)
perm = rng.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]

X = iris.data
y = iris.target


def initialize_tree() -> Tree:
    tree: Tree = Tree(5, np.zeros(1, dtype=np.int), 1)
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
    gt = GeneticTree(change_feature=1, max_depth=1)
    gt.fit(X, y)
    forest_before = gt.genetic_processor.forest
    print(f'Features of tree before mutation: {forest_before.trees[0].feature}')
    forest_after = gt.genetic_processor.mutator.mutate(forest_before)
    print(f'Features of tree in previous forest: {forest_before.trees[0].feature}')
    print(f'Features of tree after mutation: {forest_after.trees[0].feature}')


if __name__ == "__main__":
    mutate_feature()
