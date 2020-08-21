from GeneticTree import GeneticTree
from timeit import timeit
from functools import partial


def test_run(name="my_name", size=100):
    gt = GeneticTree()
    tc = gt.genetic_processor.tree_container
    tc.trees[0].test_function_with_args(bytearray(name, encoding="UTF-8"), size, 1)
    tc.function_to_test_nogil()


if __name__ == "__main__":
    time = timeit(partial(test_run, "one", 100), number=1)
    print(time)
