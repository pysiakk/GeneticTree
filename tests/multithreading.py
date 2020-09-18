from genetic_tree import GeneticTree
from tree.tree import Tree

import numpy as np
from timeit import timeit
from functools import partial
import multiprocessing
from enum import Enum


class ThreadType(Enum):
    multiprocessing = 1,
    nogil = 2,
    nogil_multi = 3,
    single = 4,
    multiprocessing_with_less_threads = 5


def test_time(size: int = 10**6, thread_type: ThreadType = ThreadType.single):
    gt = GeneticTree()
    forest = gt.genetic_processor.forest
    tc = Tree(5, 3, 10)
    if thread_type is ThreadType.multiprocessing:
        threads = []
        for i in range(100):
            t = multiprocessing.Process(target=tc.time_test, args=(size,))
            t.start()
            threads.append(t)
        for thread in threads:
            thread.join()
    elif thread_type is ThreadType.multiprocessing_with_less_threads:
        threads = []
        for i in range(10):
            t = multiprocessing.Process(target=tc.time_test2, args=(size,))
            t.start()
            threads.append(t)
        for thread in threads:
            thread.join()
    elif thread_type is ThreadType.nogil_multi:
        threads = []
        for i in range(100):
            t = multiprocessing.Process(target=tc.time_test_nogil, args=(size,))
            t.start()
            threads.append(t)
        for thread in threads:
            thread.join()
    elif thread_type is ThreadType.single:
        for i in range(100):
            target = tc.time_test(size)


if __name__ == "__main__":
    for power in range(5, 7):
        iterations: int = 10 ** power
        multi_time = timeit(partial(test_time, iterations, ThreadType.multiprocessing), number=100)
        multi_less_threads_time = timeit(partial(test_time, iterations, ThreadType.multiprocessing_with_less_threads), number=100)
        nogil_multi_time = timeit(partial(test_time, iterations, ThreadType.nogil_multi), number=100)
        single_time = timeit(partial(test_time, iterations, ThreadType.single), number=100)
        print(f'Multiprocessing time: {multi_time:0.04f} for 10^{power} iterations')
        print(f'Multiprocessing with ten time less threads than previous time: {multi_less_threads_time:0.04f} for 10^{power} iterations')
        print(f'Nogil with multiprocessing time: {nogil_multi_time:0.04f} for 10^{power} iterations')
        print(f'Single thread time: {single_time:0.04f} for 10^{power} iterations')
