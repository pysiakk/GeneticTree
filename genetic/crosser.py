

class Crosser:
    """
    Class responsible for crossing random individuals to get new ones
    """

    def __init__(self, cross_probability=0.05, **kwargs):
        self.cross_probability: float = cross_probability

    def set_params(self, cross_probability=None):
        if cross_probability is not None:
            self.cross_probability = cross_probability

    def cross_population(self):
        #TODO
        pass
