from tree.forest import Forest
from tree.tree import Tree
from tree.crosser import TreeCrosser
import numpy as np


class Crosser:
    """
    Class responsible for crossing random individuals to get new ones
    """

    """
    Cross probability means:
    For each individual A there is a cross_probability chance that
    We will find another random individual B and create 
    new individual C by crossing A with B
    Conclusion: There should be around 
    {(1+cross_probability) * population_size} individuals after crossing
    """
    def __init__(self, cross_probability: float = 0.05, **kwargs):
        self.cross_probability: float = cross_probability

    def set_params(self, cross_probability: float = None):
        if cross_probability is not None:
            self.cross_probability = cross_probability

    def cross_population(self, forest: Forest):
        crosser: TreeCrosser = TreeCrosser()

        trees_number: int = forest.current_trees
        current_trees_number: int = trees_number

        for first_parent_id in range(trees_number):
            first_parent: Tree = forest.trees[first_parent_id]
            if np.random.rand() < self.cross_probability:
                # find second parent
                second_parent_id: int = self.get_second_parent(trees_number, first_parent_id)
                second_parent: Tree = forest.trees[second_parent_id]

                # create child and register it in forest
                child: Tree = crosser.cross_trees(first_parent, second_parent)
                forest.trees[current_trees_number] = child
                current_trees_number += 1
        forest.current_trees = current_trees_number

    """
    Function to choose another individual from uniform distribution of other individuals
    """
    @staticmethod
    def get_second_parent(trees_number: int, first_parent_id: int):
        second_parent: int = np.random.randint(0, trees_number-1)
        if second_parent >= first_parent_id:
            second_parent += 1
        return second_parent
