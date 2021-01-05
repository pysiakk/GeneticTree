from tests.utils_testing import *


def test_builder_tree_size():
    for initial_depth in range(5, 0, -1):
        tree: Tree = initialize_iris_tree()
        tree.resize_by_initial_depth(initial_depth)
        full_tree_builder(tree, initial_depth)
        assert tree.node_count == tree.feature.shape[0] == tree.threshold.shape[0] == 2 ** (initial_depth+1) - 1


@pytest.mark.parametrize("function,features_assertion,threshold_assertion",
                         [(mutate_random_node,     10,  0),
                          (mutate_random_feature,  10, 10),
                          (mutate_random_threshold, 0, 10),
                          (mutate_random_class,    10,  0)])
def test_mutator(function: callable, features_assertion: int, threshold_assertion: int):
    trees = build_trees(3, 10)
    not_same_features: int = 0
    not_same_thresholds: int = 0
    for tree in trees:
        feature = np.array(tree.feature)
        threshold = np.array(tree.threshold)
        function(tree)
        not_same_features += assert_array_not_the_same_in_at_most_one_index(feature, tree.feature)
        not_same_thresholds += assert_array_not_the_same_in_at_most_one_index(threshold, tree.threshold)
    assert not_same_features >= features_assertion
    assert not_same_thresholds >= threshold_assertion


def assert_array_not_the_same_in_at_most_one_index(array, other) -> int:
    counter: int = 0
    for before, after in zip(array, other):
        if before != after:
            counter += 1
    assert counter <= 1
    return counter


def test_crossing_one_leaf():
    trees = build_trees(2, 2)
    trees[0].initialize_observations()
    trees[1].initialize_observations()
    cross_trees(trees[0], trees[1], 0, 6)


def test_crossing_only_from_second_parent():
    trees = build_trees(2, 2)
    trees[0].initialize_observations()
    trees[1].initialize_observations()
    cross_trees(trees[0], trees[1], 0, 0)


@pytest.mark.parametrize("first, second",
                         [(6, 1), (6, 0), (0, 0), (0, 1), (1, 6), (1, 3),
                          (1, 0), (6, 3), (3, 0), (3, 1), (3, 2), (3, 3), (3, 6)])
def test_crossing(first, second):
    trees = build_trees(2, 2)
    trees[0].initialize_observations()
    trees[1].initialize_observations()
    cross_trees(trees[0], trees[1], first, second)


def test_crosser():
    trees = build_trees(2, 2)
    trees[0] = trees[0], trees[0].feature, trees[0].threshold
    trees[1] = trees[1], trees[1].feature, trees[1].threshold
    tree: Tree = cross_trees(trees[0][0], trees[1][0], 2, 0)
    new_features = np.append(np.append(np.append(trees[0][1][0:2], np.array([trees[1][1][6], trees[0][1][3], trees[0][1][4]])),
                                       np.array([trees[1][1][2], trees[1][1][0], trees[1][1][5]])),
                             np.array([trees[1][1][1], trees[1][1][4], trees[1][1][3]]))
    new_depth = np.array([0, 1, 3, 2, 2, 2, 1, 3, 2, 3, 3])
    assert_array_equal(new_features, tree.feature)
    assert_array_equal(new_depth, tree.nodes_depth)


def test_independence_of_created_trees_by_crosser(crosses: int = 10, mutations: int = 10):
    trees = build_trees(1, 2)
    trees[0] = trees[0], trees[0].feature, trees[0].threshold
    trees[1] = trees[1], trees[1].feature, trees[1].threshold

    # cross tree many times with the same tree
    tree: Tree = cross_trees(trees[0][0], trees[1][0], 1, 0)
    for i in range(1, crosses):
        tree = cross_trees(trees[0][0], tree, 1, 0)

    # check if crossing is proper
    new_features = np.repeat(np.array([[trees[0][1][0], trees[0][1][2]]]).transpose(), crosses, axis=1).reshape(crosses*2, order='F')
    new_features[1] = trees[0][1][0]
    new_features[2] = trees[0][1][2]
    new_features = np.append(new_features, np.array([trees[1][1][0], trees[1][1][2], trees[1][1][1]]))
    assert_array_equal(new_features, tree.feature)

    # each mutation should mutate maximum one place in genom
    # if there are no two pointers for exactly the same node it should not mutate more than one place in genom
    old_features = np.array(tree.feature)
    for i in range(mutations):
        mutate_random_node(tree)
        new_features = np.array(tree.feature)
        assert_array_not_the_same_in_at_most_one_index(new_features, old_features)
        old_features = new_features


def test_tree_pickling():
    tree: Tree = build_trees(10, 1)[0]
    depth = tree.depth
    feature = tree.feature
    node_count = tree.node_count
    bytes_io = io.BytesIO()
    pickle.dump(tree, bytes_io)
    bytes_io.seek(0)
    tree = pickle.load(bytes_io)
    assert_array_equal(feature, tree.feature)
    assert depth == tree.depth
    assert node_count == tree.node_count


def test_tree_copying():
    tree: Tree = build_trees(10, 1)[0]
    tree_copied: Tree = copy_tree(tree)
    assert tree.node_count == tree_copied.node_count
    assert_array_equal(tree.X, tree_copied.X)
    assert_array_equal(tree.y, tree_copied.y)
    assert_array_equal(tree.thresholds, tree_copied.thresholds)
    assert id(tree.node_count) != id(tree_copied.node_count)
    assert id(tree.X) == id(tree_copied.X)
    # following lines don't have to be true
    # because two memoryviews of the same object can have different ids
    # assert id(tree.y) == id(tree_copied.y)
    # assert id(tree.thresholds) == id(tree_copied.thresholds)


def test_independence_of_copied_tree_():
    tree: Tree = build_trees(10, 1)[0]
    test_independence_of_copied_tree(tree)


# ==============================================================================
# Tree functions
# ==============================================================================

# ++++++++++++++++++++++++++
# Prediction
# ++++++++++++++++++++++++++

def build_simple_tree(threshold, class_in_leaf):
    tree: Tree = initialize_iris_tree()
    # tree, parent, is_left, feature, threshold, depth
    test_add_node(tree, TREE_UNDEFINED, 0, 2, 3.0, 1)
    # tree, parent, is_left, class, depth
    test_add_leaf(tree, 0, 1, 1, 2)

    test_add_node(tree, 0, 0, 2, threshold, 2)
    test_add_leaf(tree, 2, 1, class_in_leaf, 3)
    test_add_leaf(tree, 2, 0, 0, 3)
    return tree


@pytest.fixture
def tree() -> Tree:
    return build_simple_tree(4.8, 0)


def test_changing_classes_to_class_most_often_occurring(tree):
    tree.initialize_observations()
    assert_array_equal(tree.feature, np.array([2, 1, 2, 0, 0]))
    tree.prepare_tree_to_prediction()
    assert_array_equal(tree.feature, np.array([2, 0, 2, 1, 2]))


@pytest.mark.parametrize("tree", [build_simple_tree(6.7, 0), build_simple_tree(6.7, 1), build_simple_tree(6.7, 2)])
def test_not_changing_class_when_two_classes_have_the_same_number_of_observations(tree):
    tree.initialize_observations()
    feature = tree.feature[3]
    tree.prepare_tree_to_prediction()
    probabilities = tree.probabilities[3, :]
    if tree.feature[3] == 1:
        assert probabilities[1] >= probabilities[2]
    if tree.feature[3] == 2:
        assert probabilities[1] <= probabilities[2]
    if feature == 1 or feature == 2:
        assert feature == tree.feature[3]


def test_tree_probabilities(tree):
    tree.initialize_observations()
    tree.prepare_tree_to_prediction()
    assert_array_almost_equal(tree.probabilities[1, :], np.array([50, 1, 0]) / 51)
    assert_array_almost_equal(tree.probabilities[3, :], np.array([0, 43, 1]) / 44)
    assert_array_almost_equal(tree.probabilities[4, :], np.array([0, 6, 49]) / 55)


def test_predict(tree):
    tree.initialize_observations()
    tree.prepare_tree_to_prediction()
    assert_array_equal(tree.test_predict(X[np.argsort(X[:, 1])][:5]), np.array([1, 1, 2, 1, 0]))


def test_predict_proba(tree):
    tree.initialize_observations()
    tree.prepare_tree_to_prediction()
    prob_1 = np.array([0, 43, 1]) / 44
    assert_array_almost_equal(tree.test_predict_proba(X[np.argsort(X[:, 1])][:5]),
                              np.stack([prob_1, prob_1, np.array([0, 6, 49]) / 55, prob_1, np.array([50, 1, 0]) / 51]))


def test_apply(tree):
    tree.initialize_observations()
    tree.prepare_tree_to_prediction()
    assert_array_equal(tree.apply(X[np.argsort(X[:, 1])][:5]), np.array([3, 3, 4, 3, 1]))


# ++++++++++++++++++++++++++
# Weights
# ++++++++++++++++++++++++++

@pytest.fixture
def tree_weighted():
    sample_weight = np.random.random(150)
    # sample_weight = np.ones(150)*3
    sample_weight = np.ascontiguousarray(sample_weight, dtype=np.float32)
    tree = Tree(np.unique(y), X, y, sample_weight, thresholds, np.random.randint(10 ** 8))
    return tree, sample_weight


def test_tree_sample_weight(tree_weighted):
    tree, sample_weight = tree_weighted
    test_add_node(tree, TREE_UNDEFINED, 0, 2, 3.0, 1)
    # tree, parent, is_left, class, depth
    test_add_leaf(tree, 0, 1, 0, 2)
    test_add_leaf(tree, 0, 0, 0, 2)

    tree.initialize_observations()
    assert_array_almost_equal(tree.proper_classified, np.sum(sample_weight[y == 0]), decimal=5)


def test_tree_sample_weight_on_bigger_tree(tree_weighted):
    tree, sample_weight = tree_weighted
    test_add_node(tree, TREE_UNDEFINED, 0, 2, 3.0, 1)
    # tree, parent, is_left, class, depth
    test_add_leaf(tree, 0, 1, 1, 2)

    test_add_node(tree, 0, 0, 1, 5.7, 2)
    test_add_leaf(tree, 2, 1, 1, 3)
    test_add_leaf(tree, 2, 0, 1, 3)

    tree.initialize_observations()
    assert_almost_equal(tree.proper_classified, np.sum(sample_weight[y == 1]), decimal=5)


# ++++++++++++++++++++++++++
# Seed
# ++++++++++++++++++++++++++

def test_tree_copied_same_seed(tree):
    tree_copied = copy_tree(tree, same_seed=1)
    assert_array_equal(tree_copied.seeds, tree.seeds)


def test_tree_copied_not_same_seed(tree):
    tree_copied = copy_tree(tree, same_seed=0)
    assert tree_copied.seeds[0] != tree.seeds[0]

