from tests.utils_testing import *


def check_thresholds_memory_usage(X, X2, y2, sample_weight2, n_trees: int = 10, n_thresholds: int = 10, depth: int = 3):
    """
    Check if tree really only gets the memory view (reference for object inside memory)
    And that tree does not copy all thresholds array
    """
    X = GeneticTree._check_X(GeneticTree(), X, True)
    thresholds = prepare_thresholds_array(n_thresholds, X)

    memory = memory_used()
    print(f"Memory after creating thresholds array {memory:0.02f}.")
    trees = []
    builder: FullTreeBuilder = FullTreeBuilder()
    start = time.time()
    for i in range(n_trees):
        tree: Tree = Tree(3,  X2, y2, sample_weight2, thresholds, np.random.randint(10**8))
        tree.resize_by_initial_depth(depth)
        builder.build(tree, 3)
        trees.append(tree)
    end = time.time()
    memory_all = memory_used()
    print(trees[np.random.randint(0, len(trees))].n_features)
    print(f"All memory {memory_all:0.02f}.")
    print(f"Memory used for trees {memory_all - memory:0.02f}.")
    print(f"Creation time {end - start}.")


def memory_used():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss/(1024**2)  # in megabytes
    # resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


# Conclusion about memory used by thresholds:
# After running this code in main (but both 'check_thresholds_memory_usage' calls in other processes):
if __name__ == "__main__":
    n_thresholds = 200000000
    # y2 = np.repeat(y, 1000, axis=0)
    # X2 = np.repeat(X, 1000, axis=0)
    # sample_weight = np.repeat(sample_weight, 1000, axis=0)
    check_thresholds_memory_usage(X, X, y, sample_weight, n_thresholds=10, n_trees=10000)
    # check_thresholds_memory_usage(X, X2, y2, sample_weight, n_thresholds=10, n_trees=10000)
    # check_thresholds_memory_usage(X, X, y, sample_weight, n_thresholds=n_thresholds, n_trees=10000)
# The memory used by trees does not depend on number of thresholds array (so not depend on the size of this array)
# Also there is not a measurable time difference
# There is also no difference if X and y are bigger or not
