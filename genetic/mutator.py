from tree.tree import Tree
import math
import numpy as np

from aenum import Enum, extend_enum


class MutationType(Enum):
    """
    MutationType is enumerator with possible mutations to use:
        Class: mutate class in random leaf
        Threshold: mutate threshold in random decision node
        Feature: mutate both feature and threshold in random decision node
        ClassOrThreshold: mutate random node \
                          if decision node then mutate threshold \
                          if leaf then mutate class

    Look at SelectionType to see how to add new MutationType
    """
    def __new__(cls, function, *args):
        obj = object.__new__(cls)
        obj._value_ = len(cls.__members__)
        obj.mutate = function
        return obj

    @staticmethod
    def add_new(name, function):
        extend_enum(MutationType, name, function)

    # after each entry should be at least delimiter
    # (also can be more arguments which will be ignored)
    # this is needed because value is callable type
    Class = Tree.mutate_random_class,
    Threshold = Tree.mutate_random_threshold,
    Feature = Tree.mutate_random_feature,
    ClassOrThreshold = Tree.mutate_random_class_or_threshold,


class Mutator:
    """
    Mutator mutate individuals.

    It provides interface to allow each type of mutation
    and to set different probability for each mutation.

    The probability means the proportion of all trees that be affected by mutation

    Args:
        mutation_prob: probability to mutate random node \
                       if decision node then mutate both feature and threshold \
                       if leaf then mutate class
        mutations_additional: list of tuples \
                              each tuple contains MutationType \
                              and probability of this MutationType
    """

    def __init__(self,
                 mutation_prob: float = 0.005,
                 mutations_additional: list = None,
                 mutation_is_replace: bool = False,
                 **kwargs):
        self.mutation_prob = mutation_prob
        self.mutation_is_replace = mutation_is_replace
        if mutations_additional is not None:
            self._check_mutations_additional_(mutations_additional)
            self.mutations_additional = mutations_additional
        else:
            self.mutations_additional = []

    def set_params(self,
                   mutation_prob: float = None,
                   mutations_additional: list = None,
                   mutation_is_replace: bool = None,
                   **kwargs):
        """
        Function to set new parameters for Mutator

        Arguments are the same as in __init__
        """
        if mutation_prob is not None:
            self.mutation_prob = mutation_prob
        if mutation_is_replace is not None:
            self.mutation_is_replace = mutation_is_replace
        if mutations_additional is not None:
            self._check_mutations_additional_(mutations_additional)
            self.mutations_additional = mutations_additional

    @staticmethod
    def _check_mutations_additional_(mutations_additional):
        # TODO: meaningful errors instead of asserts
        assert isinstance(mutations_additional, list)
        for element in mutations_additional:
            assert isinstance(element[0], MutationType)
            assert isinstance(element[1], float)

    def mutate(self, trees):
        """
        It mutates all trees based on params

        First it mutate random node with probability mutation_prob
        Then for each pair (MutationType, prob) inside
        additional_mutation list it mutates MutationType with prob probability

        Args:
            trees: List with all trees to mutate

        Returns:
            mutated_trees:
        """
        mutated_population = self._mutate_by_mutation_type_(trees, None, self.mutation_prob)
        for elem in self.mutations_additional:
            mutated_population += self._mutate_by_mutation_type_(trees, elem[0], elem[1])
        return mutated_population

    def _mutate_by_mutation_type_(self, trees, mutation_type: MutationType, prob: float):
        """
        It mutate all trees by function with prob probability

        Args:
            trees: List with all trees to mutate
            mutation_type: MutationType
            prob: The probability that each tree will be mutated

        Returns:
            trees: New created trees that was mutated
        """
        new_created_trees = []
        trees_number: int = len(trees)
        tree_ids: np.array = self._get_random_trees_(trees_number, prob)
        for tree_id in tree_ids:
            tree: Tree = trees[tree_id]
            if self.mutation_is_replace:
                self._run_mutation_function_(tree, mutation_type)
            else:
                # TODO: function to intelligent copy tree (not deepcopy X, y, thresholds)
                # tree = copy_tree(tree)
                self._run_mutation_function_(tree, mutation_type)
                new_created_trees.append(tree)
        return new_created_trees

    @staticmethod
    def _run_mutation_function_(tree: Tree, mutation_type: MutationType):
        """
        Run proper mutation based on mutation_type argument.

        Args:
            tree: Tree
            mutation_type: MutationType
        """
        if mutation_type is None:
            tree.mutate_random_node()
        else:
            mutation_type.mutate(tree)

    @staticmethod
    def _get_random_trees_(n_trees: int, probability: float) -> np.array:
        """
        Warning:
            It don't use normal probability of choosing each tree

            It brings random ceil(n_trees * probability) indices

        Args:
            n_trees: Number of trees in the forest
            probability: Probability of choosing each individual

        Returns:
            np.array with indices
        """
        return np.random.choice(n_trees, math.ceil(n_trees * probability), replace=False)