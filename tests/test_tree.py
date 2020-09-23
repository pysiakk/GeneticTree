from tests.set_up_variables_and_imports import *
from genetic_tree import GeneticTree

n_thresholds: int = 10
forest: Forest = Forest(1, 2, n_thresholds)
X, y = GeneticTree.check_input(X, y)
forest.set_X_y(X, y)
forest.initialize_population(1)
thresholds = forest.thresholds


def test_builder_tree_size():
    builder: FullTreeBuilder = FullTreeBuilder(1)
    for i in range(5, 0, -1):
        builder.initial_depth = i
        tree: Tree = Tree(X.shape[1], np.unique(y).shape[0], thresholds, 3)
        builder.build(tree, X, y)
        assert tree.node_count == tree.feature.shape[0] == tree.threshold.shape[0] == 2 ** (i+1) - 1


def build(depth: int = 1, n_trees: int = 10):
    builder: FullTreeBuilder = FullTreeBuilder(depth)
    trees = []
    for i in range(n_trees):
        tree: Tree = Tree(X.shape[1], np.unique(y).shape[0], thresholds, depth)
        builder.build(tree, X, y)
        trees.append((tree, np.array(tree.feature), np.array(tree.threshold)))
    return trees


def test_mutator(function=Tree.mutate_random_feature, features_assertion: int = 10, threshold_assertion: int = 10):
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
    assert not_same_thresholds >= threshold_assertion


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
    new_features = np.append(np.append(np.append(trees[0][1][0:2], np.array([trees[0][1][4], trees[0][1][3], trees[1][1][0]])),
                                       np.array([trees[1][1][2], trees[1][1][6], trees[1][1][5]])),
                             np.array([trees[1][1][1], trees[1][1][4], trees[1][1][3]]))
    new_depth = np.array([0, 1, 2, 2, 1, 2, 3, 3, 2, 3, 3])
    assert_array_equal(new_features, tree.feature)
    assert_array_equal(new_depth, tree.nodes_depth)


def test_independence_of_created_trees_by_crosser(crosses: int = 3, mutations: int = 10):
    trees = build(1, 10)
    crosser: TreeCrosser = TreeCrosser()

    # cross tree many times with the same tree
    tree: Tree = crosser._cross_trees(trees[0][0], trees[1][0], 1, 0)
    for i in range(1, crosses):
        tree = crosser._cross_trees(trees[0][0], tree, 1, 0)

    # check if crossing is proper
    new_features = np.repeat(np.array([[trees[0][1][0], trees[0][1][2]]]).transpose(), crosses, axis=1).reshape(crosses*2, order='F')
    new_features = np.append(new_features, np.array([trees[1][1][0], trees[1][1][2], trees[1][1][1]]))
    assert_array_equal(new_features, tree.feature)

    # each mutation should mutate maximum one place in genom
    # if there are no two pointers for exactly the same node it should not mutate more than one place in genom
    old_features = np.array(tree.feature)
    for i in range(mutations):
        tree.mutate_random_node()
        new_features = np.array(tree.feature)
        assert_array_not_the_same_in_one_index(new_features, old_features)
        old_features = new_features


def test_observation_creation():
    trees = build(2, 10)
    tree: Tree = trees[0][0]
    print("\n Observation existence test: ")
    node_id: int = 6
    for k, val in tree.observations.items():
        node_id = k
        print(f'Node id: {k}, observations assigned: {len(val)}')
    observation: Observation = tree.observations[node_id][0]
    print(f'Observation id: {observation.observation_id}')
    print(f'Last node id: {observation.last_node_id}')
    print(f'Proper class: {observation.proper_class}')
    print(f'Current class: {observation.current_class}')


def test_observations_reassigning():
    trees = build(2, 10)
    tree: Tree = trees[0][0]
    for i in range(10):
        tree.mutate_random_node()
    print(f'Observations not assigned just after mutation: {len(tree.observations[-1])}')
    tree.assign_all_not_registered_observations(X)
    assert len(tree.observations[-1]) == 0


if __name__ == "__main__":
    test_builder_tree_size()
    assertion_mutator: int = 10
    test_mutator(Tree.mutate_random_node, assertion_mutator, 0)
    test_mutator(Tree.mutate_random_class, assertion_mutator, 0)
    test_mutator(Tree.mutate_random_feature, assertion_mutator, assertion_mutator)
    test_mutator(Tree.mutate_random_threshold, 0, assertion_mutator)
    test_crosser()
    test_independence_of_created_trees_by_crosser(10, 10)
    test_observation_creation()
    test_observations_reassigning()
