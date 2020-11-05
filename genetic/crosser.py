from tree.tree import Tree
from tree.crosser import TreeCrosser
import numpy as np


class Crosser:
    """
    Crosser is responsible for crossing random individuals to get new ones

    Args:
        cross_prob: The chance that each tree will be selected as first parent.
        is_cross_both: If cross first parent with second and second with first \
                       or only first with second
        is_replace_old: If replace old tree(s) by child(ren) or not

    For each tree selected with cross_prob chance there will be found second
    random parent.
    """

    def __init__(self,
                 cross_prob: float = 0.93,
                 is_cross_both: bool = True, is_replace_old: bool = True,
                 **kwargs):
        self.cross_prob: float = cross_prob
        self.is_cross_both: bool = is_cross_both
        self.is_replace_old: bool = is_replace_old

    def set_params(self,
                   cross_prob: float = None,
                   is_cross_both: bool = None, is_replace_old: bool = None,
                   **kwargs):
        """
        Function to set new parameters for Crosser

        Arguments are the same as in __init__
        """
        if cross_prob is not None:
            self.cross_prob = cross_prob
        if is_cross_both is not None:
            self.is_cross_both = is_cross_both
        if is_replace_old is not None:
            self.is_replace_old = is_replace_old

    def cross_population(self, trees):
        """
        It goes through trees inside forest and adds new trees to forest based
        on cross probability

        Args:
            forest: Container with all trees
        """
        crosser: TreeCrosser = TreeCrosser()

        trees_number: int = len(trees)

        for first_parent_id in range(trees_number):
            first_parent: Tree = trees[first_parent_id]
            if np.random.rand() < self.cross_prob:
                # find second parent
                second_parent_id: int = self.get_second_parent(trees_number, first_parent_id)
                second_parent: Tree = trees[second_parent_id]

                # create child and register it in forest
                children: Tree[:] = crosser.cross_trees(first_parent, second_parent, int(self.is_cross_both))

                # TODO change below lines to more complicated way that should be less time consuming
                # During copying nodes from first tree copy also all observations dict
                # and replace observations below changed node as NOT_REGISTERED
                # Then after completion of all tree only need to run assign_all_not_registered_observations
                children[0].initialize_observations()
                if self.is_cross_both:
                    children[1].initialize_observations()

                if self.is_replace_old:
                    trees[first_parent_id] = children[0]
                    if self.is_cross_both:
                        trees[second_parent_id] = children[1]
                else:
                    trees.append(children[0])
                    if self.is_cross_both:
                        trees.append(children[1])

    @staticmethod
    def get_second_parent(n_trees: int, first_parent_id: int) -> int:
        """
        Function to choose another individual (to cross with) from uniform
        distribution of other individuals

        Args:
            n_trees: Number of trees in the forest
            first_parent_id: Id of first chosen parent

        Returns:
            Id of second parent such that it is a number from 0 to n_trees - 1 other than first parent id
        """
        second_parent: int = np.random.randint(0, n_trees-1)
        if second_parent >= first_parent_id:
            second_parent += 1
        return second_parent
