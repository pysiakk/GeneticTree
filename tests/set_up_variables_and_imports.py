import os
os.chdir("../")
import io

from tree.tree import Tree
from tree.builder import Builder, FullTreeBuilder
from tree.crosser import cross_trees, test_cross_trees

from sklearn import datasets
import numpy as np
import pytest

from sklearn.utils._testing import assert_allclose
from sklearn.utils._testing import assert_array_equal
from sklearn.utils._testing import assert_array_almost_equal
from sklearn.utils._testing import assert_almost_equal
from sklearn.utils._testing import assert_warns
from sklearn.utils._testing import assert_warns_message
from sklearn.utils._testing import create_memmap_backed_data
from sklearn.utils._testing import ignore_warnings

# load iris dataset and randomly permute it (example from sklearn.tree.test.test_tree)
iris = datasets.load_iris()
rng = np.random.RandomState(1)
perm = rng.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]

X = iris.data
y = iris.target
y = np.ascontiguousarray(y, dtype=np.intp)
