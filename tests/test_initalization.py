from tests.set_up_variables_and_imports import *
np.random.seed(121333322)


def initialize_random(X, y, thresholds):
    initializer = Initializer(n_trees=5, initial_depth=5, initialization_type=InitializationType.Random)
    return initializer.initialize(X=X, y=y, threshold=thresholds)


def initialize_half(X, y, thresholds):
    initializer = Initializer(n_trees=20, initial_depth=5, initialization_type=InitializationType.Half)
    return initializer.initialize(X=X, y=y, threshold=thresholds)


def initialize_split(X, y, thresholds):
    initializer = Initializer(n_trees=20, initial_depth=5, initialization_type=InitializationType.Split, split_prob=0.7)
    return initializer.initialize(X=X, y=y, threshold=thresholds)


if __name__ == '__main__':
    X = GeneticTree._check_X_(GeneticTree(), X, True)
    thresholds = prepare_thresholds_array(10, X)
    trees1 = initialize_random(X, y, thresholds)
    trees2 = initialize_half(X, y, thresholds)
    trees3 = initialize_split(X, y, thresholds)

    print("\nFull initialization:")
    for tree in trees1:
        print(tree.nodes_depth)

    print("\nHalf-by-half initialization:")
    for tree in trees2:
        print(tree.nodes_depth)

    print("\nSplit initialization:")
    for tree in trees3:
        print(tree.nodes_depth)
