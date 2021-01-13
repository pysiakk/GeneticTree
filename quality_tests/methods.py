import copy
import numpy as np
from genetic_tree.genetic_tree import GeneticTree
from sklearn import datasets

iris = datasets.load_iris()
rng = np.random.RandomState(1)
perm = rng.permutation(iris.target.size)

X = iris.data[perm]
y = iris.target[perm]
y = np.ascontiguousarray(y, dtype=np.intp)

X = GeneticTree._check_X(GeneticTree(), X, True)


def test_over_params(X: list, y: list, iterate_over_1: str, iterate_params_1: list, json_path, **kwargs):
    json = "init_json"  # TODO do it properly
    data = []
    for iter_1 in range(len(iterate_params_1)):
        this_kwargs = copy.copy(kwargs)
        this_kwargs[iterate_over_1] = iterate_params_1[iter_1]
        for X_i, y_i in zip(X, y):
            gt = GeneticTree(**this_kwargs)
            gt.fit(X_i, y_i)
            json += str(gt.acc_best)  # TODO do it properly
            data.append(gt.acc_best)  # only test how it works
    # TODO save json
    return data


# TODO: crete plot by loading json with metadata
def plot_json(json_path, x_axis, y_axis, colors):
    pass


if __name__ == "__main__":
    data = test_over_params([X], [y], "max_iter", [1, 2, 3], "file_path.json")
    print(data)
    data = test_over_params([X], [y], "cross_prob", [0.5, 0.8], "file_path.json", max_iterations=5)
    print(data)
