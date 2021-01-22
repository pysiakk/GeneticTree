[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

[![example workflow name](https://github.com/pysiakk/GeneticTree/workflows/GeneticTree/badge.svg)](https://github.com/pysiakk/GeneticTree/actions?query=workflow%3AGeneticTree)
![codecov.io](https://codecov.io/github/pysiakk/GeneticTree/coverage.svg?branch=master)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/pysiakk/GeneticTree/blob/master/LICENSE)

# Genetic Tree

The main objective of the package is to allow creating decision trees that are better in some aspects than trees made by greedy algorithms.

The creation of trees is made by genetic algorithm.
In order to achive as fast as possible evolution of trees the most time consuming components are wrtitten in Cython.
Also there are implemented mechanisms for using old trees to create new ones without need to classify all observations from beggining (currently in developmnet).
There is planned to allow multithreading evolution.

The created trees should have smaller sizes with comparable accuracy to the trees made by greedy algorithms.

Project is currently in development (before first version).
The first working official version should be developed in the January 2021 (with documentation and installation by pip).

# Installation

To download the latest official release of the package use a pip command below:
```bash
pip install genetic-tree
```

# Usage

Example usage:
```python
from genetic_tree import GeneticTree
from sklearn import datasets

iris = datasets.load_iris()       # get iris data

gt = GeneticTree()
gt.fit(iris.data, iris.target)
y_pred = gt.predict(iris.data)    # it is recommended to predict on another subset of data than training
```
The `y_pred` contains an array with classes predicted by the `GeneticTree`

## License

The work is a banchelor thesis on Warsaw University of Technology. Commercial use forbidden.

High-level interface of package is inspired by sklearn (https://github.com/scikit-learn/scikit-learn).
Especially there are methods like: fit(), predict(), predict_proba(), apply(), set_params(), check_X(), check_input() which are inspired and / or copied from sklearn.

A low-level interface is inspired by sklearn decision_tree. The structure of tree (tree/tree.pyx) and some utils (tree/\_utils.pyx) were copied from sklearn tree (https://github.com/scikit-learn/scikit-learn/tree/master/sklearn/tree).
