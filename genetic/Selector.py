from enum import Enum, auto

class SelectionType(Enum):
    RankSelection = auto()

class Metric(Enum):
    Accuracy = auto()

class Selector:
    """
    Class responsible for selecting best individuals from population
    First of all it evaluates each individual and give them a score
    Then it selects best population

    There are plenty of possible metrics:
    - accuracy

    Possible selection policies:
    - best n

    There is also elitarysm, which means that top k individuals are selected
    before selection policy is used
    """

    def __init__(self, n_trees=1000, selection_type=SelectionType.RankSelection, metric=Metric.Accuracy, elitarysm=5):
        self.n_trees: int = n_trees
        self.selection_type: SelectionType = selection_type
        self.metric: Metric = metric
        self.elitarysm: int = elitarysm

    def set_params(self, n_trees=None, selection_type=None, metric=None, elitarysm=None):
        if n_trees is not None:
            self.n_trees = n_trees
        if selection_type is not None:
            self.selection_type = selection_type
        if metric is not None:
            self.metric = metric
        if elitarysm is not None:
            self.elitarysm = elitarysm

    def select(self):
        self.__evaluate__()
        self.__leave_best_population__()

    def __evaluate__(self):
        #TODO
        pass

    def __leave_best_population__(self):
        #TODO
        pass


