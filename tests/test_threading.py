import time
from threading import Thread

from genetic_tree import GeneticTree
from tests.set_up_variables_and_imports import *


def check_creating_trees_with_many_threads(n_trees: int = 10, n_jobs: int = 4, depth: int = 3):
    """
    Check if using many threads to initializing trees save time
    """
    gt = GeneticTree(initial_depth=1, remove_variables=False, remove_other_trees=False, max_iterations=1)
    gt.fit(X, y)
    trees = []
    start = time.time()
    threads = []
    for i in range(n_jobs):
        process = Thread(target=create_trees_in_one_thread, args=[n_trees//n_jobs, gt.forest.thresholds, depth, trees])
        process.start()
        threads.append(process)
    for process in threads:
        process.join()
    end = time.time()
    print(f"Creation time {end - start}.")


def create_trees_in_one_thread(n_trees, thresholds, depth, trees):
    builder: FullTreeBuilder = FullTreeBuilder(depth)
    for i in range(n_trees):
        tree: Tree = Tree(3, X, y, thresholds)
        tree.resize_by_initial_depth(depth)
        builder.build(tree)
        # tree.initialize_observations(X, y)
        trees.append(tree)


if __name__ == "__main__":
    for depth in [3, 7, 10, 15, 18]:
        print(f"\n Depth {depth}.")
        check_creating_trees_with_many_threads(n_trees=100, n_jobs=1, depth=depth)
        check_creating_trees_with_many_threads(n_trees=100, n_jobs=4, depth=depth)
        check_creating_trees_with_many_threads(n_trees=100, n_jobs=100, depth=depth)
