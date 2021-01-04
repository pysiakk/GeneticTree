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
                 # most important params
                 n_thresholds: int = 10,
                 n_trees: int = 400,
                 max_iter: int = 500,
                 cross_prob: float = 0.6,
                 mutation_prob: float = 0.4,
                 initialization_type: InitializationType = InitializationType.Split,
                 metric: Metric = Metric.AccuracyMinusDepth,
                 selection_type: SelectionType = SelectionType.StochasticUniform,
                 n_elitism: int = 3,
                 early_stopping: bool = False,
                 n_iter_no_change: int = 100,

                 # additional genetic algorithm params
                 cross_is_both: bool = True,
                 mutations_additional: list = None,
                 mutation_is_replace: bool = False,
                 initial_depth: int = 1,
                 split_prob: float = 0.7,
                 n_leaves_factor: float = 0.0001,
                 depth_factor: float = 0.01,
                 tournament_size: int = 3,
                 is_leave_selected_parents: bool = False,

                 # technical params
                 random_state: int = None,
                 is_save_metrics: bool = True,
                 is_keep_last_population: bool = False,
                 is_remove_variables: bool = True,

                 # TODO: params not used yet:
                 verbose: bool = True,
                 n_jobs: int = -1,
                 max_depth: int = 20,
                 **kwargs
                 ):

        if random_state is not None:
            np.random.seed(random_state)

        kwargs = vars()
        kwargs.pop('self')
        kwargs.pop('random_state')
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

        self._is_save_metrics = is_save_metrics
        self.acc_mean = []
        self.acc_best = []
        self.n_leaves_mean = []
        self.n_leaves_best = []
        self.depth_mean = []
        self.depth_best = []
        self.metric_best = []
        self.metric_mean = []

        self._is_keep_last_population = is_keep_last_population
        self._is_remove_variables = is_remove_variables
        self._is_leave_selected_parents = is_leave_selected_parents

        self._trees = None
        self._best_tree = None

        self._n_thresholds = n_thresholds
        self._n_features = None
        self._can_predict = False

    @staticmethod
    def _is_any_arg_none(possible_nones, **kwargs):
        for k, val in kwargs.items():
            if val is None:
                if k not in possible_nones:
                    return k
        return False

    def set_params(self, is_keep_last_population: bool = None, is_remove_variables: bool = None):
        # TODO add all params

        kwargs = vars()
        kwargs.pop('self')

        self.initializer.set_params(**kwargs)
        self.mutator.set_params(**kwargs)
        self.crosser.set_params(**kwargs)
        self.selector.set_params(**kwargs)
        self.evaluator.set_params(**kwargs)
        self.stop_condition.set_params(**kwargs)
        if is_keep_last_population is not None:
            self._is_keep_last_population = is_keep_last_population
        if is_remove_variables is not None:
            self._is_remove_variables = is_remove_variables

    def fit(self, X, y, *args, sample_weight: np.array = None, check_input: bool = True, **kwargs):
        self._can_predict = False
        self.set_params(**kwargs)
        X, y, sample_weight = self._check_input(X, y, sample_weight, check_input)
        self._prepare_new_training(X, y, sample_weight)
        self._growth_trees()
        self._prepare_to_predict()

    def _prepare_new_training(self, X, y, sample_weight):
        self.stop_condition.reset_private_variables()

        thresholds = prepare_thresholds_array(self._n_thresholds, X)
        if self._trees is None:  # when previously trees was removed
            self._trees = self.initializer.initialize(X, y, sample_weight, thresholds)

    def _growth_trees(self):
        offspring = self._trees
        trees_metrics = self.evaluator.evaluate(offspring)
        self._append_metrics(offspring)

        while not self.stop_condition.stop(trees_metrics):
            elite = self.selector.get_elite_population(offspring, trees_metrics)
            selected_parents = self.selector.select(offspring, trees_metrics)
            mutated_population = self.mutator.mutate(selected_parents)
            crossed_population = self.crosser.cross_population(selected_parents)

            # offspring based on elite parents from previous
            # population, and trees made by mutation and crossing
            offspring = mutated_population + crossed_population
            if self._is_leave_selected_parents:
                offspring += selected_parents
            else:
                offspring += elite

            trees_metrics = self.evaluator.evaluate(offspring)
            self._append_metrics(offspring)

        self._trees = offspring

    def _prepare_to_predict(self):
        self._prepare_best_tree_to_prediction()
        if not self._is_keep_last_population:
            self._trees = None
            if self._is_remove_variables:
                self._best_tree.remove_variables()
        elif self._is_remove_variables:
            for tree in self._trees:
                tree.remove_variables()
        self._can_predict = True

    def _prepare_best_tree_to_prediction(self):
        best_tree_index: int = self.evaluator.get_best_tree_index(self._trees)
        self._best_tree = self._trees[best_tree_index]
        self._best_tree.prepare_tree_to_prediction()
    
    def _append_metrics(self, trees):
        if self._is_save_metrics:
            best_tree_index = self.evaluator.get_best_tree_index(trees)
            accuracies = self.evaluator.get_accuracies(trees)
            self.acc_best.append(accuracies[best_tree_index])
            self.acc_mean.append(np.mean(accuracies))
            depths = self.evaluator.get_depths(trees)
            self.depth_best.append(depths[best_tree_index])
            self.depth_mean.append(np.mean(depths))
            n_leaves = self.evaluator.get_n_leaves(trees)
            self.n_leaves_best.append(n_leaves[best_tree_index])
            self.n_leaves_mean.append(np.mean(n_leaves))
            metrics = self.evaluator.evaluate(trees)
            self.metric_best.append(metrics[best_tree_index])
            self.metric_mean.append(np.mean(metrics))

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
        self._check_is_fitted()
        X = self._check_X(X, check_input)
        return self._best_tree.predict(X)

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
        self._check_is_fitted()
        X = self._check_X(X, check_input)
        return self._best_tree.predict_proba(X)

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
        self._check_is_fitted()
        X = self._check_X(X, check_input)
        return self._best_tree.apply(X)

    def _check_is_fitted(self):
        if not self._can_predict:
            raise Exception('Cannot predict. Model not prepared.')

    def _check_input(self, X, y, sample_weight, check_input: bool):
        """
        Check if X and y have proper dtype and have the same number of observations

        Args:
            X: np.array or scipy.sparse_matrix of size observations x features
            y: np.array with proper classes of observations
            check_input: if check X and y or not

        Returns:
            X and y in proper format
        """
        X = self._check_X(X, check_input)

        if check_input:
            if y.dtype != SIZE or not y.flags.contiguous:
                y = np.ascontiguousarray(y, dtype=SIZE)

            if sample_weight is None:
                sample_weight = np.ones(y.shape[0], dtype=np.float32)
            else:
                if sample_weight.shape[0] != y.shape[0]:
                    raise ValueError(f"y and sample_weight should have the same "
                                     f"number of observations. Weights "
                                     f"have {sample_weight.shape[0]} observations "
                                     f"and y have {y.shape[0]} observations.")
                if sample_weight.dtype != np.float32 or not sample_weight.flags.contiguous:
                    sample_weight = np.ascontiguousarray(sample_weight, dtype=np.float32)

        if y.shape[0] != X.shape[0]:
            raise ValueError(f"X and y should have the same number of "
                             f"observations. X have {X.shape[0]} observations "
                             f"and y have {y.shape[0]} observations.")

        return X, y, sample_weight

    def _check_X(self, X, check_input: bool):
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
        if self._n_features is None:
            self._n_features = n_features
        elif self._n_features != n_features:
            raise ValueError(f"Number of features of the model must match the "
                             f"input. Model n_features is {self._n_features} "
                             f"and input n_features is {n_features}.")

        return X
