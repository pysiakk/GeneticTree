import numpy as np

from genetic.initializer import Initializer
from genetic.initializer import InitializationType
from genetic.mutator import Mutator
from genetic.crosser import Crosser
from genetic.selector import Selector
from genetic.stop_condition import StopCondition
from tree.forest import Forest
from scipy.sparse import issparse

from numpy import float32 as DTYPE
from numpy import float64 as DOUBLE


class GeneticTree:
    """
    High level interface possible to use like scikit-learn class
    """

    def __init__(self, n_trees: int = 1000, max_trees: int = 2000, n_thresholds: int = 10,
                 initial_depth: int = 3, initialization_type: InitializationType = InitializationType.Random,
                 is_feature: bool = True, feature_prob: float = 0.05,
                 is_threshold: bool = True, threshold_prob: float = 0.05,
                 is_class: bool = True, class_prob: float = 0.05,
                 cross_prob: float = 0.05,
                 max_iterations: int = 1000,
                 max_iterations_without_improvement: int = 100, use_without_improvement: bool = False,
                 remove_other_trees: bool = True, remove_variables: bool = True,
                 ):
        # TODO check: if any of parameters is None -> write warning / throw error
        # TODO write all kwargs
        self.genetic_processor = \
            GeneticProcessor(n_trees=n_trees, max_trees=max_trees, n_thresholds=n_thresholds,
                             initial_depth=initial_depth, initialization_type=initialization_type,
                             is_feature=is_feature, feature_prob=feature_prob,
                             is_threshold=is_threshold, threshold_prob=threshold_prob,
                             is_class=is_class, class_prob=class_prob,
                             cross_prob=cross_prob,
                             max_iterations=max_iterations, use_without_improvement=use_without_improvement,
                             max_iterations_without_improvement=max_iterations_without_improvement,
                             remove_variables=remove_variables, remove_other_trees=remove_other_trees,
                             )
        self.__can_predict__ = False

    def set_params(self):
        #TODO write all kwargs
        self.genetic_processor.set_params()

    def fit(self, X, y, check_input: bool = True, **kwargs):
        self.__can_predict__ = False
        self.genetic_processor.set_params(**kwargs)
        if check_input:
            X, y = self.check_input(X, y)
        self.genetic_processor.prepare_new_training(X, y)
        self.genetic_processor.growth_trees()
        self.__prepare_to_predict__()

    def __prepare_to_predict__(self):
        self.genetic_processor.prepare_to_predict()
        self.__can_predict__ = True

    def predict(self, X):
        if not self.__can_predict__:
            raise Exception('Cannot predict. Model not prepared.')
        X = self.check_input(X)
        return self.genetic_processor.predict(X)

    @staticmethod
    def check_input(X, y):
        """
        Check if X and y have proper dtype and have the same number of observations

        Args:
            X: np.array or scipy.sparse_matrix of size observations x features
            y: np.array with proper classes of observations

        Returns:
            X and y in proper format
        """
        X = GeneticTree.check_X(X)

        if y.dtype != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        if y.shape[0] != X.shape[0]:
            raise ValueError("X and y should have the same number of observations")

        return X, y

    @staticmethod
    def check_X(X):
        """
        Checks if X has proper dtype
        If not it return proper X

        Args:
            X: np.array or scipy.sparse_matrix of size observations x features

        Returns:
            Converted X
        """
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

        return X


class GeneticProcessor:
    """
    Low level interface responsible for communication between all genetic classes
    """

    def __init__(self, remove_other_trees: bool = True, remove_variables: bool = True, **kwargs):
        self.initializer = Initializer(**kwargs)
        self.mutator = Mutator(**kwargs)
        self.crosser = Crosser(**kwargs)
        self.selector = Selector(**kwargs)
        self.stop_condition = StopCondition(**kwargs)
        self.forest = Forest(kwargs["n_trees"], kwargs["max_trees"], kwargs["n_thresholds"])
        self.remove_other_trees = remove_other_trees
        self.remove_variables = remove_variables

    def set_params(self, remove_other_trees: bool = None, remove_variables: bool = None, **kwargs):
        self.initializer.set_params(**kwargs)
        self.mutator.set_params(**kwargs)
        self.crosser.set_params(**kwargs)
        self.selector.set_params(**kwargs)
        self.stop_condition.set_params(**kwargs)
        if remove_other_trees is not None:
            self.remove_other_trees = remove_other_trees
        if remove_variables is not None:
            self.remove_variables = remove_variables

    def prepare_new_training(self, X, y):
        self.forest.set_X_y(X, y)
        self.stop_condition.reset_private_variables()
        self.initializer.initialize(self.forest)
        self.selector.set_n_observations(y.shape[0])

    def growth_trees(self):
        while not self.stop_condition.stop():
            self.mutator.mutate(self.forest)
            self.crosser.cross_population(self.forest)
            self.selector.select(self.forest)

    def prepare_to_predict(self):
        best_tree_index: int = self.selector.get_best_tree_index(self.forest)
        self.forest.prepare_best_tree_to_prediction(best_tree_index)
        if self.remove_other_trees:
            self.forest.remove_other_trees()
        if self.remove_variables:
            self.forest.remove_unnecessary_variables()

    def predict(self, X):
        return self.forest.predict(X)
