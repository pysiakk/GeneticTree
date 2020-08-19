

class Mutator:
    """
    Class responsible for mutation of individuals
    """

    def __init__(self, mutate_attributes=True, change_attribute=0.05, mutate_thresholds=True, change_threshold=0.05):
        self.mutate_attributes: bool = mutate_attributes
        self.change_attribute: float = change_attribute
        self.mutate_thresholds: bool = mutate_thresholds
        self.change_threshold: float = change_threshold

    def set_params(self, mutate_attributes=None, change_attribute=None, mutate_thresholds=None, change_threshold=None):
        if mutate_attributes is not None:
            self.mutate_attributes = mutate_attributes
        if change_attribute is not None:
            self.change_attribute = change_attribute
        if mutate_thresholds is not None:
            self.mutate_thresholds = mutate_thresholds
        if change_threshold is not None:
            self.change_threshold = change_threshold

    def mutate(self):
        #TODO
        pass
