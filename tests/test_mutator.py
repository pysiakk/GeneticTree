from tests.utils_testing import *

# ==============================================================================
# Mutation functions (if it will be standalone)
# ==============================================================================


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
                          (0.2, [(MutationType.Feature, 0.3)], True),
                          (0.5, [(MutationType.Feature, 0.4), (MutationType.Threshold, 0.2)], False),
                          ])
def test_mutator_init(prob, mutations_additional, is_replace):
    mutator = Mutator(prob, mutations_additional, is_replace)
    assert prob == mutator.mutation_prob
    assert is_replace == mutator.mutation_is_replace
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
    mutator.set_params(mutation_is_replace=is_replace)
    assert mutator.mutation_is_replace == is_replace


@pytest.mark.parametrize("is_replace", ["string", [True]])
def test_set_is_replace_wrong_type(mutator, is_replace):
    with pytest.raises(TypeError):
        mutator.set_params(mutation_is_replace=is_replace)


@pytest.mark.parametrize("mutations_additional", [[], [(MutationType.Feature, 0.4), (MutationType.Threshold, 0.2)]])
def test_set_additional_mutations(mutator, mutations_additional):
    mutator.set_params(mutations_additional=mutations_additional)
    assert mutator.mutations_additional == mutations_additional


@pytest.mark.parametrize("mutations_additional", [True, "string",
                                                  [("string", 0.4)],
                                                  [(MutationType.Feature, "string")]])
def test_set_additional_mutations_wrong_type(mutator, mutations_additional):
    with pytest.raises(TypeError):
        mutator.set_params(mutations_additional=mutations_additional)


def test_set_new_mutation(mutator):
    MutationType.add_new("NewMutation", lambda x: x)
    mutator.set_params(mutations_additional=[(MutationType.NewMutation, 0.2)])
    assert str(type(mutator.mutations_additional[0][0])) == str(MutationType)


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


@pytest.mark.parametrize("mutation_type, mutation_function",
                         [(MutationType.Feature, mutate_random_feature),
                          (None, mutate_random_node)])
def test_run_mutation_function(trees, mutation_type, mutation_function):
    tree = trees[0]
    tree_mutated = copy_tree(tree, same_seed=1)
    Mutator._run_mutation_function(tree_mutated, mutation_type)
    mutation_function(tree)
    assert_array_equal(tree.feature, tree_mutated.feature)
    assert_array_equal(tree.threshold, tree_mutated.threshold)


@pytest.mark.parametrize("mutation_type, mutation_function",
                         [(MutationType.Feature, mutate_random_feature),
                          (None, mutate_random_node)])
def test_mutate_by_mutation_type_prob_one_replace(trees, mutation_type, mutation_function):
    trees = trees[:1]
    trees_copied = [copy_tree(tree, same_seed=1) for tree in trees]
    Mutator._mutate_by_mutation_type(Mutator(mutation_is_replace=True), trees_copied, mutation_type, 1)
    for tree, tree_mutated in zip(trees, trees_copied):
        mutation_function(tree)
        assert_array_equal(tree.feature, tree_mutated.feature)
        assert_array_equal(tree.threshold, tree_mutated.threshold)


@pytest.mark.parametrize("mutation_type, mutation_function",
                         [(MutationType.Feature, mutate_random_feature),
                          (None, mutate_random_node)])
def test_mutate_by_mutation_type_prob_one_not_replace(trees, mutation_type, mutation_function):
    trees = trees[:1]
    trees_copied = [copy_tree(tree, same_seed=1) for tree in trees]
    trees_copied_random_seed = [copy_tree(tree, same_seed=0) for tree in trees]
    trees_mutated = Mutator._mutate_by_mutation_type(Mutator(mutation_is_replace=False), trees_copied, mutation_type, 1)
    for tree, tree_mutated in zip(trees_copied_random_seed, trees_mutated):
        mutation_function(tree)
        assert_array_equal(tree.feature, tree_mutated.feature)
        assert_array_equal(tree.threshold, tree_mutated.threshold)


@pytest.mark.parametrize("mutation_type, mutation_function",
                         [(MutationType.Feature, mutate_random_feature),
                          (None, mutate_random_node)])
def test_mutate_by_mutation_type_prob_zero(trees, mutation_type, mutation_function):
    trees_copied = [copy_tree(tree, same_seed=1) for tree in trees]
    Mutator._mutate_by_mutation_type(Mutator(), trees_copied, mutation_type, 0)
    for tree, tree_mutated in zip(trees, trees_copied):
        assert_array_equal(tree.feature, tree_mutated.feature)
        assert_array_equal(tree.threshold, tree_mutated.threshold)
