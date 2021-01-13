from ..tree.tree import Tree, copy_tree
import math
import numpy as np

from aenum import Enum, extend_enum
from ..tree.mutator import mutate_random_node, mutate_random_class_or_threshold
from ..tree.mutator import mutate_random_feature, mutate_random_threshold
from ..tree.mutator import mutate_random_class


class Mutation(Enum):
    """
    Mutation is enumerator with possible mutations to use:
        Class: mutate class in random leaf
        Threshold: mutate threshold in random decision node
        Feature: mutate both feature and threshold in random decision node
        ClassOrThreshold: mutate random node \
                          if decision node then mutate threshold \
                          if leaf then mutate class

    Look at Selection to see how to add new Mutation
    """
    def __new__(cls, function, *args):
        obj = object.__new__(cls)
        obj._value_ = len(cls.__members__)
        obj.mutate = function
        return obj

    @staticmethod
    def add_new(name, function):
        extend_enum(Mutation, name, function)

    # after each entry should be at least delimiter
    # (also can be more arguments which will be ignored)
    # this is needed because value is callable type
    Class = mutate_random_class,
    Threshold = mutate_random_threshold,
    Feature = mutate_random_feature,
    ClassOrThreshold = mutate_random_class_or_threshold,


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
                              each tuple contains Mutation \
                              and probability of this Mutation
        mutation_replace: if new trees should replace previous or should \
                             previous trees be modified directly
    """

    def __init__(self,
                 mutation_prob: float = 0.4,
                 mutations_additional: list = None,
                 mutation_replace: bool = False,
                 **kwargs):
        self.mutation_prob = self._check_mutation_prob(mutation_prob)
        self.mutation_replace = self._check_mutation_replace(mutation_replace)
        if mutations_additional is not None:
            self.mutations_additional = self._check_mutations_additional(mutations_additional)
        else:
            self.mutations_additional = []

    def set_params(self,
                   mutation_prob: float = None,
                   mutations_additional: list = None,
                   mutation_replace: bool = None,
                   **kwargs):
        """
        Function to set new parameters for Mutator

        Arguments are the same as in __init__
        """
        if mutation_prob is not None:
            self.mutation_prob = self._check_mutation_prob(mutation_prob)
        if mutation_replace is not None:
            self.mutation_replace = self._check_mutation_replace(mutation_replace)
        if mutations_additional is not None:
            self.mutations_additional = self._check_mutations_additional(mutations_additional)

    @staticmethod
    def _check_mutation_prob(mutation_prob, error_name: str = "mutation_prob"):
        if type(mutation_prob) is not float and type(mutation_prob) is not int:
            raise TypeError(f"{error_name}: {mutation_prob} should be "
                            f"float or int. Instead it is {type(mutation_prob)}")
        if mutation_prob <= 0:
            mutation_prob = 0
        if mutation_prob >= 1:
            mutation_prob = 1
        return mutation_prob
    
    @staticmethod
    def _check_mutation_replace(mutation_replace):
        if type(mutation_replace) is not bool:
            raise TypeError(f"mutation_replace: {mutation_replace} should be "
                            f"bool. Instead it is {type(mutation_replace)}")
        return mutation_replace

    @staticmethod
    def _check_mutations_additional(mutations_additional):
        if not isinstance(mutations_additional, list):
            raise TypeError(f"mutations_additional: {mutations_additional} is "
                            f"not type list")
        for i in range(len(mutations_additional)):
            element = mutations_additional[i]
            # comparison of strings because after using Mutation.add_new() Mutation is reference to other class
            if str(type(element[0])) != str(Mutation):
                raise TypeError(f"Mutation inside mutations additional: "
                                f"{element[0]} is not a Mutation")
            error_name = f"Mutation probability inside mutations additional for {element[0]}"
            element = element[0], Mutator._check_mutation_prob(element[1], error_name)
            mutations_additional[i] = element
        return mutations_additional

    def mutate(self, trees):
        """
        It mutates all trees based on params

        First it mutate random node with probability mutation_prob
        Then for each pair (Mutation, prob) inside
        additional_mutation list it mutates Mutation with prob probability

        Args:
            trees: List with all trees to mutate

        Returns:
            mutated_trees:
        """
        mutated_population = self._mutate_by_mutation(trees, None, self.mutation_prob)
        for elem in self.mutations_additional:
            mutated_population += self._mutate_by_mutation(trees, elem[0], elem[1])
        return mutated_population

    def _mutate_by_mutation(self, trees, mutation: Mutation, prob: float):
        """
        It mutate all trees by function with prob probability

        Args:
            trees: List with all trees to mutate
            mutation: Mutation
            prob: The probability that each tree will be mutated

        Returns:
            trees: New created trees that was mutated
        """
        new_created_trees = []
        trees_number: int = len(trees)
        tree_ids: np.array = self._get_random_trees(trees_number, prob)
        for tree_id in tree_ids:
            tree: Tree = trees[tree_id]
            if self.mutation_replace:
                self._run_mutation_function(tree, mutation)
            else:
                tree = copy_tree(tree)
                self._run_mutation_function(tree, mutation)
                new_created_trees.append(tree)
        return new_created_trees

    @staticmethod
    def _run_mutation_function(tree: Tree, mutation: Mutation):
        """
        Run proper mutation based on mutation argument.

        Args:
            tree: Tree
            mutation: Mutation
        """
        if mutation is None:
            mutate_random_node(tree)
        else:
            mutation.mutate(tree)

    @staticmethod
    def _get_random_trees(n_trees: int, probability: float) -> np.array:
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
