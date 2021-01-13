.. _examples:

==============
Usage examples
==============

Train model
===========

First we need to construct a classifiers' object.
During construction we can set parameters (more :doc:`GeneticTree API </api/genetic_tree>`).
In this example we only set mutation_prob

.. code-block:: python

    gt = GeneticTree(mutation_prob=0.3)

Then we can fit a model with dataset X and vector of proper classes y (both numpy array).

.. code-block:: python

  gt.fit(X, y)

Predict
=======

Then model can be used to predict on new dataset:

.. code-block:: python

  y_pred = gt.predict(X_new)

The returned value is one dimensional numpy array.

Predict probabilities
=====================

Other possibility is to use model to predict class probabilities:

.. code-block:: python

  y_probabilities = gt.predict_proba(X_new)

The returned value is two dimensional numpy array of shape [number of observations in X_new, number of classes].


Training logs
=============

During training model saves some metrics about individuals.
Those metrics can be get from model after training.

.. code-block:: python

    gt = GeneticTree()
    gt.fit(X, y)
    gt.acc_best
    gt.acc_mean
    gt.depth_best
    gt.depth_mean
    gt.n_leaves_best
    gt.n_leaves_mean

There are 6 metrics. The accuracy, depth and leaves number of the best individual
(by accuracy) and a mean of all individuals. All of them stores a list of values
after initialization and after each iteration.

Finding the best parameter
==========================

Using those metrics user can find optimal parameters by the training. They can
create a plot how the metric is looking after each epoch. But it can be also
simpler case. For example this code snippet shows how to get cross_prob which
gives the best accuracy after training (on training dataset).

.. code-block:: python

    cross_prob = [0.2, 0.4, 0.6, 0.8]
    accuracy_best = []
    for i in range(len(cross_prob)):
        gt = GeneticTree(max_iter=10, cross_prob=cross_prob[i])
        gt.fit(X, y)
        accuracy_best.append(gt.acc_best[-1])
    best_accuracy_id = np.argmax(np.array(accuracy_best))
    print(f"Best accuracy is for cross prob: {cross_prob[best_accuracy_id]}")

Stream training
===============

Sometimes it is useful to retrain previous model on new data. It is possible by
using partial_fit.

.. code-block:: python
   :linenos:
   :emphasize-lines: 1,11

    gt = GeneticTree(max_iter=10, keep_last_population=True)
    gt.fit(X, y)
    while True:
        X_new = get_X_from_last_day()
        y_new = get_y_from_last_day()
        weights = np.ones(y.shape[0])
        weights_new = np.ones(y_new.shape[0]) * 2
        weights_all = np.concatenate([weights, weights_new])
        X = np.concatenate([X, X_new])
        y = np.concatenate([y, y_new])
        gt.partial_fit(X, y, sample_weight=weights_all)
        wait_for_next_day()

To retrain dataset first we need to set keep_last_population=True and the model
will keep all individuals after fit ended. Then each retraining should be done
using partial_fit(), because the population will not be initialized from
beginning. Also in retraining on streaming data it is useful to get new data
bigger weights and pass them using sample_weight. For example each day there
are new data and during training this new data gets weight 2 but data from
previous days get only weight 1. So the new data is more valuable for
individuals to fit (partial_fit because of retraining).
