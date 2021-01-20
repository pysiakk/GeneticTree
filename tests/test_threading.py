from tests.utils_testing import *
import pandas as pd


def check_creating_trees_with_many_threads(X, n_trees: int = 10, n_jobs: int = 4, depth: int = 3):
    """
    Check if using many threads to initializing trees save time
    """
    X = GeneticTree._check_X(GeneticTree(), X, True)
    thresholds = prepare_thresholds_array(10, X)

    trees = []
    start = time.time()
    threads = []
    for i in range(n_jobs):
        process = Thread(target=create_trees_in_one_thread, args=[n_trees//n_jobs, thresholds, depth, trees])
        process.start()
        threads.append(process)
    for process in threads:
        process.join()
    end = time.time()
    print(f"Creation time {end - start}.")


def create_trees_in_one_thread(n_trees, thresholds, depth, trees):
    for i in range(n_trees):
        tree: Tree = Tree(3, X, y, sample_weight, thresholds, np.random.randint(10**8))
        tree.resize_by_initial_depth(depth)
        full_tree_builder(tree, depth)
        # tree.initialize_observations(X, y)
        trees.append(tree)


n_trees = 60

plants = pd.read_csv("https://www.openml.org/data/get_csv/1592285/phpoOxxNn.csv")
plants_X = np.array(plants.iloc[:, :plants.shape[1] - 1])
plants_y = np.array(plants["Class"])
X_big, y_big = plants_X, plants_y
for i in range(10):
    X_big, y_big = np.concatenate([X_big, plants_X]), np.concatenate([y_big, plants_y])


@pytest.mark.parametrize("n_jobs, max_iter, X, y",
                         [
                             ([1, 2], 100, X, y),
                             ([1, 2], 100, np.concatenate([X, X, X, X, X, X, X, X, X, X]),
                              np.concatenate([y, y, y, y, y, y, y, y, y, y])),
                             ([1, 2], 1000, plants_X, plants_y),
                             ([3, 4], 1000, plants_X, plants_y),
                             ([1, 2], 100, plants_X, plants_y),
                             ([1, 2], 100, X_big, y_big),
                             ([3, 4], 100, X_big, y_big),
                         ])
# def test_seed(n_jobs=[1, 1], max_iter=3, X=X, y=y):
def multithreading_seed_test(n_jobs, max_iter, X, y):
    """
    Assert that algorithm will create the same best trees with set seed
    """
    start = time.time()
    seed = np.random.randint(0, 10**8)
    gt = GeneticTree(random_state=seed, n_trees=n_trees, max_iter=max_iter, n_jobs=n_jobs[0])
    gt.fit(X, y)
    tree: Tree = gt._best_tree
    print(f"\n Time: {time.time() - start}, N_jobs = {n_jobs[0]}, X.shape = {X.shape}, iters = {max_iter}")

    start = time.time()
    gt2 = GeneticTree(random_state=seed, n_trees=n_trees, max_iter=max_iter, n_jobs=n_jobs[1])
    gt2.fit(X, y)
    tree2: Tree = gt2._best_tree
    print(f"Time: {time.time() - start}, N_jobs = {n_jobs[1]}, X.shape = {X.shape}, iters = {max_iter}")

    assert_trees_equal(tree, tree2)


# if __name__ == "__main__":
#     for depth in [3, 7, 10, 15, 18]:
#         print(f"\n Depth {depth}.")
#         check_creating_trees_with_many_threads(X, n_trees=100, n_jobs=1, depth=depth)
#         check_creating_trees_with_many_threads(X, n_trees=100, n_jobs=4, depth=depth)
#         check_creating_trees_with_many_threads(X, n_trees=100, n_jobs=100, depth=depth)
