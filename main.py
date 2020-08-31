from GeneticTree import GeneticTree
from timeit import timeit
from functools import partial
import multiprocessing


def test_run(name="my_name", size=100, multi=True):
    gt = GeneticTree()
    tc = gt.genetic_processor.tree_container
    if multi:
        threads = []
        for i in range(100):
            t = multiprocessing.Process(target=tc.trees[0].test_function_with_args, args=(bytearray(name, encoding="UTF-8"), size, 1))
            t.start()
            threads.append(t)
        for thread in threads:
            thread.join()
    else:
        for i in range(100):
            target = tc.trees[0].test_function_with_args(bytearray(name, encoding="UTF-8"), size, 1)


if __name__ == "__main__":
    multi_time = timeit(partial(test_run, "one", 10000000000, True), number=1)
    single_time = timeit(partial(test_run, "one", 10000000000, True), number=1)
    print(f'Multiprocessing time: {multi_time:0.04f}')
    print(f'Single thread time: {single_time:0.04f}')
