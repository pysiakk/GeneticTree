from tree.tree import Tree
from tree.crosser import cross_trees
import numpy as np


class Crosser:
    """
    Crosser is responsible for crossing random individuals to get new ones

    Args:
        cross_prob: The chance that each tree will be selected as first parent.
        cross_is_both: If cross first parent with second and second with first \
                       or only first with second

    For each tree selected with cross_prob chance there will be found second
    random parent.
    """

    def __init__(self,
                 cross_prob: float = 0.93,
                 cross_is_both: bool = True,
                 **kwargs):
        self.cross_prob: float = cross_prob
        self.cross_is_both: bool = cross_is_both

    def set_params(self,
                   cross_prob: float = None,
                   cross_is_both: bool = None,
                   **kwargs):
        """
        Function to set new parameters for Crosser

        Arguments are the same as in __init__
        """
        if cross_prob is not None:
            self.cross_prob = cross_prob
        if cross_is_both is not None:
            self.cross_is_both = cross_is_both

    def cross_population(self, trees):
        """
        It goes through trees inside forest and adds new trees to forest based
        on cross probability

        Args:
            trees: List with all trees to apply crossing
        """
        new_created_trees = []

        trees_number: int = len(trees)

        first_parents_indices: np.array = self._get_random_trees_(trees_number, self.cross_prob)
        second_parents_indices: np.array = self._get_second_parents_(trees_number, first_parents_indices)

        for i in range(first_parents_indices.shape[0]):
            first_parent: Tree = trees[first_parents_indices[i]]
            second_parent: Tree = trees[second_parents_indices[i]]

            first_node_id: int = first_parent.get_random_node()
            second_node_id: int = second_parent.get_random_node()

            # create children and append them to list
            new_created_trees.append(cross_trees(first_parent, second_parent, first_node_id, second_node_id))
            if self.cross_is_both:
                new_created_trees.append(cross_trees(second_parent, first_parent, second_node_id, first_node_id))

        return new_created_trees

    @staticmethod
    def _get_random_trees_(n_trees: int, probability: float) -> np.array:
        """
        Args:
            n_trees: Number of trees
            probability: Probability of choosing each individual

        Returns:
            np.array with indices
        """
        indices: np.array = np.arange(0, n_trees)
        random_values: np.array = np.random.random(n_trees)
        return indices[random_values < probability]

    @staticmethod
    def _get_second_parents_(n_trees: int, first_parents: np.array) -> np.array:
        """
        Function to choose another individual (to cross with) from uniform
        distribution of other individuals
        It choose second individual for each individual in first_parents array

        Args:
            first_parents: Array with indices of first parents
            n_trees: Number of trees

        Returns:
            Array of ids of second parents such that each id is a number from 0 \
            to n_trees - 1 other than first parent id
        """
        second_parents: np.array = np.random.randint(0, n_trees-1, first_parents.shape[0])
        second_parents += (second_parents >= first_parents)
        return second_parents
