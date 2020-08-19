import math

class StopCondition:
    """
    Class responsible for checking stop condition in genetic algorithm

    Implemented conditions:
    maximum number of iterations

    In development:
    maximum number of iterations without improvement of best individual
    """

    def __init__(self, max_iterations=1000, max_iterations_without_improvement=100, use_without_improvement=False):
        self.max_iterations: int = max_iterations
        self.max_iterations_without_improvement: int = max_iterations_without_improvement
        self.use_without_improvement: bool = use_without_improvement

        #private variables
        self.reset_private_variables()

    def set_params(self, max_iterations=None, max_iterations_without_improvement=None, use_without_improvement=None):
        if max_iterations is not None:
            self.max_iterations = max_iterations
        if max_iterations_without_improvement is not None:
            self.max_iterations_without_improvement = max_iterations_without_improvement
        if use_without_improvement is not None:
            self.use_without_improvement = use_without_improvement

    def reset_private_variables(self):
        self.actual_iteration: int = 1
        self.best_result: float = -math.inf
        self.best_result_iteration: int = 1

    def stop(self) -> bool:
        if self.actual_iteration > self.max_iterations:
            return True

        if self.use_without_improvement:
            #TODO
            pass

        self.actual_iteration += 1
        return False
