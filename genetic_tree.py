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
from tree.forest import Forest
from scipy.sparse import issparse

from numpy import float32 as DTYPE
from numpy import intp as SIZE


class GeneticTree:
    """
    High level interface possible to use like scikit-learn class
    """

    def __init__(self,
                 n_trees: int = 200, max_trees: int = 600, n_thresholds: int = 10,
                 initial_depth: int = 1, initialization_type: InitializationType = InitializationType.Random,
                 is_feature: bool = False, feature_prob: float = 0.005,
                 is_threshold: bool = False, threshold_prob: float = 0.005,
                 is_class: bool = False, class_prob: float = 0.005,
                 is_node: bool = False, node_prob: float = 0.005,
                 is_class_or_threshold: bool = True, class_or_threshold_prob: float = 0.005,
                 cross_prob: float = 0.93,
                 is_cross_both: bool = True, is_replace_old: bool = False,
                 max_iterations: int = 200,
                 max_iterations_without_improvement: int = 100, use_without_improvement: bool = False,
                 selection_type: SelectionType = SelectionType.RankSelection,
                 metric: Metric = Metric.AccuracyBySize, size_coef: int = 1000,
                 elitarysm: int = 5,
                 remove_other_trees: bool = True, remove_variables: bool = True,
                 seed: int = None,

                 # TODO: params not used yet:
                 verbose: bool = True, n_jobs: int = -1,
                 ):

        if seed is not None:
            np.random.seed(seed)
        else:
            seed = 0    # because it will return ValueError inside
                        # statement `if none_arg:`

        kwargs = vars()
        kwargs.pop('self')
        none_arg = self.is_any_arg_none(**kwargs)
        if none_arg:
            raise ValueError(f"The argument {none_arg} is None. "
                             f"GeneticTree does not support None arguments.")

        self.initializer = Initializer(**kwargs)
        self.mutator = Mutator(**kwargs)
        self.crosser = Crosser(**kwargs)
        self.selector = Selector(**kwargs)
        self.evaluator = Evaluator(**kwargs)
        self.stop_condition = StopCondition(**kwargs)
        self.forest = Forest(n_trees, max_trees, n_thresholds)

        self.remove_other_trees = remove_other_trees
        self.remove_variables = remove_variables
        self._n_features_ = None
        self._can_predict_ = False

    @staticmethod
    def is_any_arg_none(**kwargs):
        for k, val in kwargs.items():
            if val is None:
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

    def fit(self, X, y, check_input: bool = True, **kwargs):
        self._can_predict_ = False
        self.set_params(**kwargs)
        X, y = self._check_input_(X, y, check_input)
        self._prepare_new_training_(X, y)
        self._growth_trees_()
        self._prepare_to_predict_()

    def _prepare_new_training_(self, X, y):
        self.forest.set_X_y(X, y)
        self.forest.prepare_thresholds_array()
        self.stop_condition.reset_private_variables()
        self.initializer.initialize(self.forest, X, y, self.forest.thresholds)

    def _growth_trees_(self):
        while not self.stop_condition.stop():
            self.mutator.mutate(self.forest)
            self.crosser.cross_population(self.forest)
            trees_metric = self.evaluator.evaluate(self.forest)
            self.selector.select(self.forest, trees_metric)

    def _prepare_to_predict_(self):
        best_tree_index: int = self.evaluator.get_best_tree_index(self.forest)
        self.forest.prepare_best_tree_to_prediction(best_tree_index)
        if self.remove_other_trees:
            self.forest.remove_other_trees()
        if self.remove_variables:
            self.forest.remove_unnecessary_variables()
        self._can_predict_ = True

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
        return self.forest.predict(X)

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
        return self.forest.best_tree.predict_proba(X)

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
        return self.forest.best_tree.apply(X)

    def _check_is_fitted_(self):
        if not self._can_predict_:
            raise Exception('Cannot predict. Model not prepared.')

    def _check_input_(self, X, y, check_input: bool):
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

        if y.shape[0] != X.shape[0]:
            raise ValueError(f"X and y should have the same number of "
                             f"observations. X have {X.shape[0]} observations "
                             f"and y have {y.shape[0]} observations.")

        return X, y

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
