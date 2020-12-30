import numpy as np

from genetic.initializer import Initializer
from genetic.initializer import InitializationType
from genetic.mutator import Mutator
from genetic.crosser import Crosser
from genetic.selector import Selector
from genetic.selector import SelectionType
from genetic.evaluator import Evaluator
from genetic.evaluator import Metric
from genetic.stop_condition import StopCondition
from tree.thresholds import prepare_thresholds_array
from scipy.sparse import issparse

from numpy import float32 as DTYPE
from numpy import intp as SIZE


class GeneticTree:
    """
    High level interface possible to use like scikit-learn class
    """

    def __init__(self,
                 n_trees: int = 400,
                 n_thresholds: int = 10,
                 # TODO: change default initialization to splitting nodes with probability
                 initial_depth: int = 1,
                 initialization_type: InitializationType = InitializationType.Random,
                 split_prob: float = 0.7,
                 mutation_prob: float = 0.4,
                 mutations_additional: list = None,
                 mutation_is_replace: bool = False,
                 cross_prob: float = 0.6,
                 cross_is_both: bool = True,
                 is_left_selected_parents: bool = False,
                 max_iterations: int = 500,
                 # TODO: params about stopping algorithm when it coverages
                 selection_type: SelectionType = SelectionType.StochasticUniform,
                 n_elitism: int = 3,
                 metric: Metric = Metric.AccuracyMinusDepth,
                 remove_other_trees: bool = True, remove_variables: bool = True,
                 seed: int = None,
                 save_metrics: bool = True,

                 # TODO: params not used yet:
                 verbose: bool = True, n_jobs: int = -1,
                 ):

        if seed is not None:
            np.random.seed(seed)

        kwargs = vars()
        kwargs.pop('self')
        kwargs.pop('seed')
        none_arg = self._is_any_arg_none(['mutations_additional'], **kwargs)
        if none_arg:
            raise ValueError(f"The argument {none_arg} is None. "
                             f"GeneticTree does not support None arguments.")

        self.initializer = Initializer(**kwargs)
        self.mutator = Mutator(**kwargs)
        self.crosser = Crosser(**kwargs)
        self.selector = Selector(**kwargs)
        self.evaluator = Evaluator(**kwargs)
        self.stop_condition = StopCondition(**kwargs)

        self._save_metrics = save_metrics
        self.acc_mean = []
        self.acc_best = []
        self.n_leaves_mean = []
        self.n_leaves_best = []
        self.depth_mean = []
        self.depth_best = []

        self.remove_other_trees = remove_other_trees
        self.remove_variables = remove_variables
        self._is_left_selected_parents_ = is_left_selected_parents

        self._trees_ = None
        self._best_tree_ = None

        self._n_thresholds_ = n_thresholds
        self._n_features_ = None
        self._can_predict_ = False

    @staticmethod
    def _is_any_arg_none(possible_nones, **kwargs):
        for k, val in kwargs.items():
            if val is None:
                if k not in possible_nones:
                    return k
        return False

    def set_params(self, remove_other_trees: bool = None, remove_variables: bool = None):
        # TODO add all params

        kwargs = vars()
        kwargs.pop('self')

        self.initializer.set_params(**kwargs)
        self.mutator.set_params(**kwargs)
        self.crosser.set_params(**kwargs)
        self.selector.set_params(**kwargs)
        self.evaluator.set_params(**kwargs)
        self.stop_condition.set_params(**kwargs)
        if remove_other_trees is not None:
            self.remove_other_trees = remove_other_trees
        if remove_variables is not None:
            self.remove_variables = remove_variables

    def fit(self, X, y, *args, weights: np.array = None, check_input: bool = True, **kwargs):
        self._can_predict_ = False
        self.set_params(**kwargs)
        X, y, weights = self._check_input(X, y, weights, check_input)
        self._prepare_new_training_(X, y, weights)
        self._growth_trees_()
        self._prepare_to_predict_()

    def _prepare_new_training_(self, X, y, weights):
        self.stop_condition.reset_private_variables()

        thresholds = prepare_thresholds_array(self._n_thresholds_, X)
        if self._trees_ is None:  # when previously trees was removed
            self._trees_ = self.initializer.initialize(X, y, weights, thresholds)

    def _growth_trees_(self):
        offspring = self._trees_
        trees_metrics = self.evaluator.evaluate(offspring)
        self._append_metrics(offspring)

        while not self.stop_condition.stop(max(trees_metrics)):
            elite = self.selector.get_elite_population(offspring, trees_metrics)
            selected_parents = self.selector.select(offspring, trees_metrics)
            mutated_population = self.mutator.mutate(selected_parents)
            crossed_population = self.crosser.cross_population(selected_parents)

            # offspring based on elite parents from previous
            # population, and trees made by mutation and crossing
            offspring = mutated_population + crossed_population
            if self._is_left_selected_parents_:
                offspring += selected_parents
            else:
                offspring += elite

            trees_metrics = self.evaluator.evaluate(offspring)
            self._append_metrics(offspring)

        self._trees_ = offspring

    def _prepare_to_predict_(self):
        self._prepare_best_tree_to_prediction_()
        if self.remove_other_trees:
            self._trees_ = None
            if self.remove_variables:
                self._best_tree_.remove_variables()
        elif self.remove_variables:
            for tree in self._trees_:
                tree.remove_variables()
        self._can_predict_ = True

    def _prepare_best_tree_to_prediction_(self):
        best_tree_index: int = self.evaluator.get_best_tree_index(self._trees_)
        self._best_tree_ = self._trees_[best_tree_index]
        self._best_tree_.prepare_tree_to_prediction()
    
    def _append_metrics(self, trees):
        # TODO should best metric be a best of all or a metric of best tree?
        if self._save_metrics:
            acc = self.evaluator.get_accuracies(trees)
            self.acc_best.append(np.max(acc))
            self.acc_mean.append(np.mean(acc))
            depth = self.evaluator.get_depths(trees)
            self.depth_best.append(np.min(depth))
            self.depth_mean.append(np.mean(depth))
            n_leaves = self.evaluator.get_n_leaves(trees)
            self.n_leaves_best.append(np.min(n_leaves))
            self.n_leaves_mean.append(np.mean(n_leaves))

    def predict(self, X, check_input=True):
        """
        Function to classify each observation in X to one class

        Args:
            X: np.array or scipy.sparse_matrix of size observations x features
            check_input: if check X or not, allows to not check input many times

        Returns:
            Array of size X.shape[0]. \n
            For each row x (observation) it classify the observation to one
            class and return this class.
        """
        self._check_is_fitted_()
        X = self._check_X_(X, check_input)
        return self._best_tree_.predict(X)

    def predict_proba(self, X, check_input=True):
        """
        Function to classify each observation in X with the probability of
        being each class

        Args:
            X: np.array or scipy.sparse_matrix of size observations x features
            check_input: if check X or not, allows to not check input many times

        Returns:
            Array of size (X.shape[0], n_classes). \n
            For each row x in X (observation) it finds the proper leaf. Then it
            returns the probability of each class based on leaf.
        """
        self._check_is_fitted_()
        X = self._check_X_(X, check_input)
        return self._best_tree_.predict_proba(X)

    def apply(self, X, check_input=True):
        """
        Return the index of the leaf that each sample is predicted as.

        Args:
            X: np.array or scipy.sparse_matrix of size observations x features
            check_input: if check X or not, allows to not check input many times

        Returns:
            Array of size X.shape[0]. \n
            For each observation x in X, return the index of the leaf x
            ends up in. Leaves are numbered within [0, node_count).
        """
        self._check_is_fitted_()
        X = self._check_X_(X, check_input)
        return self._best_tree_.apply(X)

    def _check_is_fitted_(self):
        if not self._can_predict_:
            raise Exception('Cannot predict. Model not prepared.')

    def _check_input(self, X, y, weights, check_input: bool):
        """
        Check if X and y have proper dtype and have the same number of observations

        Args:
            X: np.array or scipy.sparse_matrix of size observations x features
            y: np.array with proper classes of observations
            check_input: if check X and y or not

        Returns:
            X and y in proper format
        """
        X = self._check_X_(X, check_input)

        if check_input:
            if y.dtype != SIZE or not y.flags.contiguous:
                y = np.ascontiguousarray(y, dtype=SIZE)

            if weights is None:
                weights = np.ones(y.shape[0], dtype=np.float32)
            else:
                if weights.shape[0] != y.shape[0]:
                    raise ValueError(f"y and weights should have the same "
                                     f"number of observations. Weights "
                                     f"have {weights.shape[0]} observations "
                                     f"and y have {y.shape[0]} observations.")
                if weights.dtype != np.float32 or not weights.flags.contiguous:
                    weights = np.ascontiguousarray(weights, dtype=np.float32)

        if y.shape[0] != X.shape[0]:
            raise ValueError(f"X and y should have the same number of "
                             f"observations. X have {X.shape[0]} observations "
                             f"and y have {y.shape[0]} observations.")

        return X, y, weights

    def _check_X_(self, X, check_input: bool):
        """
        Checks if X has proper dtype
        If not it return proper X

        Args:
            X: np.array or scipy.sparse_matrix of size observations x features
            check_input: if check X or not

        Returns:
            Converted X
        """
        if check_input:
            if issparse(X):
                X = X.tocsr()
                X.sort_indices()

                if X.data.dtype != DTYPE:
                    X.data = np.ascontiguousarray(X.data, dtype=DTYPE)

                if X.indices.dtype != np.int32 or X.indptr.dtype != np.int32:
                    raise ValueError("No support for np.int64 index based "
                                     "sparse matrices")

            elif X.dtype != DTYPE:
                X = np.ascontiguousarray(X, dtype=DTYPE)

        # even if check_input is false it should check n_features of X
        n_features = X.shape[1]
        if self._n_features_ is None:
            self._n_features_ = n_features
        elif self._n_features_ != n_features:
            raise ValueError(f"Number of features of the model must match the "
                             f"input. Model n_features is {self._n_features_} "
                             f"and input n_features is {n_features}.")

        return X
