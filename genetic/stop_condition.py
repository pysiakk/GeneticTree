import math
import statistics

class StopCondition:
    """
    Class responsible for checking stop condition in genetic algorithm

    Implemented conditions:
    maximum number of iterations

    In development:
    maximum number of iterations without improvement of best individual
    """

    def __init__(self, max_iterations: int = 1000,
                 max_iterations_without_improvement: int = 100, use_without_improvement: bool = False,
                 **kwargs):
        self.max_iterations: int = max_iterations
        self.max_iterations_without_improvement: int = max_iterations_without_improvement
        self.use_without_improvement: bool = use_without_improvement
        self.best_metric_hist = []

        #private variables
        self.reset_private_variables()

    def set_params(self, max_iterations: int = None,
                   max_iterations_without_improvement: int = None, use_without_improvement: bool = None,
                   **kwargs):
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

    def stop(self, score: float = None) -> bool:
        if self.actual_iteration > self.max_iterations:
            return True

        if self.use_without_improvement:
            if len(self.best_metric_hist) < self.max_iterations_without_improvement:
                self.best_metric_hist.append(score)
            else:
                self.best_metric_hist.append(score)
                self.best_metric_hist.pop(0)
            if self.best_metric_hist[self.max_iterations_without_improvement-1] <= statistics.median(self.best_metric_hist[:self.max_iterations_without_improvement-1]):
                return True

        self.actual_iteration += 1
        return False
