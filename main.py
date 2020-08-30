from GeneticTree import GeneticTree
from timeit import timeit
from functools import partial
import multiprocessing

def test_run(name="my_name", size=100):
    gt = GeneticTree()
    tc = gt.genetic_processor.tree_container
    # threads = []
    # for i in range(10):
    #     t = multiprocessing.Process(target=tc.trees[0].test_function_with_args, args=(bytearray(name, encoding="UTF-8"), size, 1))
    #     t.start()
    #     threads.append(t)
    # for thread in threads:
    #     thread.join()
    for i in range(10):
        target = tc.trees[0].test_function_with_args(bytearray(name, encoding="UTF-8"), size, 1)


if __name__ == "__main__":
    time = timeit(partial(test_run, "one", 10000000000), number=1)
    print(time)
