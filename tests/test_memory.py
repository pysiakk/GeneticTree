import psutil
import time

from genetic_tree import GeneticTree
from tests.set_up_variables_and_imports import *


def check_thresholds_memory_usage(n_trees: int = 10, n_thresholds: int = 10, depth: int = 3):
    """
    Check if tree really only gets the memory view (reference for object inside memory)
    And that tree does not copy all thresholds array
    """
    gt = GeneticTree(initial_depth=1, remove_variables=False, remove_other_trees=False, max_iterations=1, n_thresholds=n_thresholds)
    gt.fit(X, y)
    memory = memory_used()
    print(f"Memory after creating thresholds array {memory:0.02f}.")
    trees = []
    builder: FullTreeBuilder = FullTreeBuilder()
    start = time.time()
    for i in range(n_trees):
        tree: Tree = Tree(4, 3, gt.forest.thresholds, depth)
        builder.build(tree, 3)
        # tree.initialize_observations(X, y)
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
    check_thresholds_memory_usage(n_thresholds=10, n_trees=10000)
    check_thresholds_memory_usage(n_thresholds=n_thresholds, n_trees=10000)
# The memory used by trees does not depend on number of thresholds array (so not depend on the size of this array)
# Also there is not a measurable time difference
