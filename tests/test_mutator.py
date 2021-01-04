from tests.utils_testing import *
from tests.test_tree import build_simple_tree


# ==============================================================================
# Mutation functions
# ==============================================================================

@pytest.fixture
def tree() -> Tree:
    return build_simple_tree(4.8, 0)


def mutate_node_test(tree: Tree, copied_tree: Tree):
    node_id = copied_tree.randint(0, 5)
    if node_id == 0 or node_id == 2:  # if decision node
        test_mutate_feature(copied_tree, node_id)
    else:
        test_mutate_class(copied_tree, node_id)

    mutate_random_node(tree)


def mutate_class_or_threshold_test(tree: Tree, copied_tree: Tree):
    node_id = copied_tree.randint(0, 5)
    if node_id == 0 or node_id == 2:  # if decision node
        test_mutate_threshold(copied_tree, node_id)
    else:
        test_mutate_class(copied_tree, node_id)

    mutate_random_class_or_threshold(tree)


def mutate_feature_test(tree: Tree, copied_tree: Tree):
    node_id = copied_tree.get_random_decision_node_test()
    new_feature_id = copied_tree.randint(0, 3)
    if new_feature_id >= copied_tree.feature[node_id]:
        new_feature_id += 1
    copied_tree.change_feature_or_class_test(node_id, new_feature_id)
    test_mutate_threshold(copied_tree, node_id, feature_changed=1)

    mutate_random_feature(tree)


def mutate_threshold_test(tree: Tree, copied_tree: Tree):
    node_id = copied_tree.get_random_decision_node_test()
    feature_id = copied_tree.feature[node_id]
    new_threshold_id = copied_tree.randint(0, copied_tree.n_thresholds - 1)
    if copied_tree.thresholds[new_threshold_id, feature_id] >= copied_tree.threshold[node_id]:
        new_threshold_id += 1
    copied_tree.change_threshold_test(node_id, copied_tree.thresholds[new_threshold_id, feature_id])

    mutate_random_threshold(tree)


def mutate_class_test(tree: Tree, copied_tree: Tree):
    node_id = copied_tree.get_random_leaf_test()
    new_class_id = copied_tree.randint(0, 2)
    if new_class_id >= copied_tree.feature[node_id]:
        new_class_id += 1
    copied_tree.change_feature_or_class_test(node_id, new_class_id)

    mutate_random_class(tree)


@pytest.mark.parametrize("function", [mutate_node_test, mutate_class_or_threshold_test,
                                      mutate_class_test, mutate_feature_test, mutate_threshold_test])
def test_mutation(tree, function):
    copied_tree = copy_tree(tree, same_seed=1)
    for i in range(100):
        seeds = [np.random.randint(0, 10**8), np.random.randint(0, 10**8),
                 np.random.randint(0, 10**8), np.random.randint(0, 10**8)]
        tree.seeds = seeds
        copied_tree.seeds = seeds
        function(tree, copied_tree)
        assert_array_equal(tree.feature, copied_tree.feature)
        assert_array_equal(tree.threshold, copied_tree.threshold)


# ==============================================================================
# Mutator
# ==============================================================================

# Base mutator
@pytest.fixture
def mutator():
    return Mutator(0.4, None, False)


@pytest.fixture
def trees():
    trees = []
    for i in range(3):
        trees.append(build_trees(5, 1)[0])
    return trees


# +++++++++++++++
# Init
# +++++++++++++++

@pytest.mark.parametrize("prob, mutations_additional, is_replace",
                         [(0.1, None, True),
                          (0.2, [(Mutation.Feature, 0.3)], True),
                          (0.5, [(Mutation.Feature, 0.4), (Mutation.Threshold, 0.2)], False),
                          ])
def test_mutator_init(prob, mutations_additional, is_replace):
    mutator = Mutator(prob, mutations_additional, is_replace)
    assert prob == mutator.mutation_prob
    assert is_replace == mutator.mutation_replace
    if mutations_additional is None:
        assert mutator.mutations_additional == []
    else:
        assert mutations_additional == mutator.mutations_additional


# +++++++++++++++
# Set params
# +++++++++++++++

@pytest.mark.parametrize("mutation_prob", [0.1, 0.2, 0.4])
def test_set_mutation_prob(mutator, mutation_prob):
    mutator.set_params(mutation_prob=mutation_prob)
    assert mutator.mutation_prob == mutation_prob


@pytest.mark.parametrize("mutation_prob", [-1, -0.1, 0])
def test_set_mutation_prob_below_0(mutator, mutation_prob):
    mutator.set_params(mutation_prob=mutation_prob)
    assert mutator.mutation_prob == 0


@pytest.mark.parametrize("mutation_prob", [1, 1.1, 10])
def test_set_mutation_prob_above_1(mutator, mutation_prob):
    mutator.set_params(mutation_prob=mutation_prob)
    assert mutator.mutation_prob == 1


@pytest.mark.parametrize("mutation_prob", ["string", [1]])
def test_set_mutation_prob_wrong_type(mutator, mutation_prob):
    with pytest.raises(TypeError):
        mutator.set_params(mutation_prob=mutation_prob)


@pytest.mark.parametrize("is_replace", [True, False])
def test_set_is_replace(mutator, is_replace):
    mutator.set_params(mutation_replace=is_replace)
    assert mutator.mutation_replace == is_replace


@pytest.mark.parametrize("is_replace", ["string", [True]])
def test_set_is_replace_wrong_type(mutator, is_replace):
    with pytest.raises(TypeError):
        mutator.set_params(mutation_replace=is_replace)


@pytest.mark.parametrize("mutations_additional", [[], [(Mutation.Feature, 0.4), (Mutation.Threshold, 0.2)]])
def test_set_additional_mutations(mutator, mutations_additional):
    mutator.set_params(mutations_additional=mutations_additional)
    assert mutator.mutations_additional == mutations_additional


@pytest.mark.parametrize("mutations_additional", [True, "string",
                                                  [("string", 0.4)],
                                                  [(Mutation.Feature, "string")]])
def test_set_additional_mutations_wrong_type(mutator, mutations_additional):
    with pytest.raises(TypeError):
        mutator.set_params(mutations_additional=mutations_additional)


def test_set_new_mutation(mutator):
    Mutation.add_new("NewMutation", lambda x: x)
    mutator.set_params(mutations_additional=[(Mutation.NewMutation, 0.2)])
    assert str(type(mutator.mutations_additional[0][0])) == str(Mutation)


# +++++++++++++++
# Set params
# +++++++++++++++

@pytest.mark.parametrize("n_trees, prob",
                         [(10, 0.3),
                          (12, 0.4),
                          (1234, 0.1)])
def test_get_random_trees(n_trees, prob):
    trees = Mutator._get_random_trees(n_trees, prob)
    assert len(trees) == math.ceil(n_trees * prob)
    assert len(trees) == len(np.unique(trees))


@pytest.mark.parametrize("mutation, mutation_function",
                         [(Mutation.Feature, mutate_random_feature),
                          (None, mutate_random_node)])
def test_run_mutation_function(trees, mutation, mutation_function):
    tree = trees[0]
    tree_mutated = copy_tree(tree, same_seed=1)
    Mutator._run_mutation_function(tree_mutated, mutation)
    mutation_function(tree)
    assert_array_equal(tree.feature, tree_mutated.feature)
    assert_array_equal(tree.threshold, tree_mutated.threshold)


@pytest.mark.parametrize("mutation, mutation_function",
                         [(Mutation.Feature, mutate_random_feature),
                          (None, mutate_random_node)])
def test_mutate_by_mutation_prob_one_replace(trees, mutation, mutation_function):
    trees = trees[:1]
    trees_copied = [copy_tree(tree, same_seed=1) for tree in trees]
    Mutator._mutate_by_mutation(Mutator(mutation_replace=True), trees_copied, mutation, 1)
    for tree, tree_mutated in zip(trees, trees_copied):
        mutation_function(tree)
        assert_array_equal(tree.feature, tree_mutated.feature)
        assert_array_equal(tree.threshold, tree_mutated.threshold)


@pytest.mark.parametrize("mutation, mutation_function",
                         [(Mutation.Feature, mutate_random_feature),
                          (None, mutate_random_node)])
def test_mutate_by_mutation_prob_one_not_replace(trees, mutation, mutation_function):
    trees = trees[:1]
    trees_copied = [copy_tree(tree, same_seed=1) for tree in trees]
    trees_copied_random_seed = [copy_tree(tree, same_seed=0) for tree in trees]
    trees_mutated = Mutator._mutate_by_mutation(Mutator(mutation_replace=False), trees_copied, mutation, 1)
    for tree, tree_mutated in zip(trees_copied_random_seed, trees_mutated):
        mutation_function(tree)
        assert_array_equal(tree.feature, tree_mutated.feature)
        assert_array_equal(tree.threshold, tree_mutated.threshold)


@pytest.mark.parametrize("mutation, mutation_function",
                         [(Mutation.Feature, mutate_random_feature),
                          (None, mutate_random_node)])
def test_mutate_by_mutation_prob_zero(trees, mutation, mutation_function):
    trees_copied = [copy_tree(tree, same_seed=1) for tree in trees]
    Mutator._mutate_by_mutation(Mutator(), trees_copied, mutation, 0)
    for tree, tree_mutated in zip(trees, trees_copied):
        assert_array_equal(tree.feature, tree_mutated.feature)
        assert_array_equal(tree.threshold, tree_mutated.threshold)


def test_mutate(trees, mutator):
    mutations_additional = [(Mutation.Feature, 1), (Mutation.Threshold, 1)]
    mutator.set_params(mutation_prob=1, mutations_additional=mutations_additional, mutation_replace=False)
    trees_mutated = mutator.mutate(trees)
    assert len(trees) * 3 == len(trees_mutated)

