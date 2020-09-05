from tests.set_up_variables_and_imports import *


def test_builder_tree_size():
    builder: FullTreeBuilder = FullTreeBuilder(1)
    for i in range(5, 0, -1):
        builder.depth = i
        tree: Tree = Tree(5, 1)
        builder.build(tree, X, y)
        assert tree.node_count == tree.feature.shape[0] == tree.threshold.shape[0] == 2 ** (i+1) - 1


def build(depth: int = 1, n_trees: int = 10):
    builder: FullTreeBuilder = FullTreeBuilder(depth)
    trees = []
    for i in range(n_trees):
        tree: Tree = Tree(X.shape[1], np.unique(y).shape[0])
        builder.build(tree, X, y)
        trees.append((tree, np.array(tree.feature), np.array(tree.threshold)))
    return trees


def test_mutator(function, features_assertion: int = 7, threshold_assertion: int = 7):
    trees = build(3, 10)
    not_same_features: int = 0
    not_same_thresholds: int = 0
    for tree, feature, threshold in trees:
        function(tree)
        not_same_features += assert_array_not_the_same_in_one_index(feature, tree.feature)
        not_same_thresholds += assert_array_not_the_same_in_one_index(threshold, tree.threshold)
    assert not_same_features >= features_assertion  # because of random mutation it can be up to 10
                                                    # but also equal 0 with low probability
    # uncomment after creating proper thresholds setting
    # assert not_same_thresholds >= threshold_assertion


def assert_array_not_the_same_in_one_index(array, other) -> int:
    counter: int = 0
    for before, after in zip(array, other):
        if before != after:
            counter += 1
    assert counter <= 1
    return counter


def test_crosser():
    trees = build(2, 10)
    crosser: TreeCrosser = TreeCrosser()
    tree: Tree = crosser._cross_trees(trees[0][0], trees[1][0], 2, 0)
    new_features = np.append(np.append(np.append(np.append(np.append(trees[0][1][0:2], [0, 0])
                                                           , trees[1][1][0]), trees[1][1][2:5])
                                       , trees[1][1][1])
                             , [0, 0])
    new_depth = np.array([0, 1, 2, 2, 1, 2, 3, 3, 2, 3, 3])
    assert_array_equal(new_features, tree.feature)
    assert_array_equal(new_depth, tree.depth)


if __name__ == "__main__":
    test_builder_tree_size()
    assertion_mutator: int = 6
    test_mutator(Tree.mutate_random_node, assertion_mutator, assertion_mutator)
    test_mutator(Tree.mutate_random_class, assertion_mutator, 0)
    test_mutator(Tree.mutate_random_feature, assertion_mutator, assertion_mutator)
    test_mutator(Tree.mutate_random_threshold, 0, assertion_mutator)
    test_crosser()
