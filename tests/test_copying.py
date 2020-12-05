from tests.utils_testing import *


def test_copy_tree(n: int = 2, depth: int = 10):
    tree: Tree = build_trees(depth, 1)[0]
    start = time.time()
    for i in range(n):
        copy_tree(tree)
    return time.time() - start


def test_crossing_tree(n: int = 2, depth: int = 10):
    tree: Tree = build_trees(depth, 1)[0]
    tree2: Tree = build_trees(depth, 1)[0]
    start = time.time()
    node_id = tree.node_count - 1
    for i in range(n):
        cross_trees(tree, tree2, node_id, node_id)
    return time.time() - start


if __name__ == "__main__":
    n = 10
    for depth in [2, 5, 7, 10, 12, 15, 18, 20]:
        print(f"\n Depth {depth}")
        copy_time = test_copy_tree(n, depth=depth)
        print(f"Copy time {copy_time}")
        cross_time = test_crossing_tree(n, depth=depth)
        print(f"Cross time {cross_time}")
        print(f"Factor = {cross_time/copy_time}")

