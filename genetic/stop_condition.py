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

    def __init__(self, n_iters: int = 500,
                 n_iters_without_improvement: int = 100, use_without_improvement: bool = False,
                 **kwargs):
        self.n_iters: int = n_iters
        self.n_iters_without_improvement: int = n_iters_without_improvement
        self.use_without_improvement: bool = use_without_improvement

        #private variables
        self.reset_private_variables()

    def set_params(self, n_iters: int = None,
                   n_iters_without_improvement: int = None, use_without_improvement: bool = None,
                   **kwargs):
        if n_iters is not None:
            self.n_iters = n_iters
        if n_iters_without_improvement is not None:
            self.n_iters_without_improvement = n_iters_without_improvement
        if use_without_improvement is not None:
            self.use_without_improvement = use_without_improvement

    def reset_private_variables(self):
        self.current_iteration: int = 1
        self.best_result: float = -math.inf
        self.best_result_iteration: int = 1
        self.best_metric_hist: list = []

    def stop(self, metrics: list = None) -> bool:
        score = max(metrics)
        if self.current_iteration > self.n_iters:
            return True

        if self.use_without_improvement:
            if len(self.best_metric_hist) < self.n_iters_without_improvement:
                self.best_metric_hist.append(score)
            else:
                self.best_metric_hist.append(score)
                self.best_metric_hist.pop(0)
                if self.best_metric_hist[self.n_iters_without_improvement - 1] \
                        <= statistics.median(self.best_metric_hist[:self.n_iters_without_improvement - 1]):
                    return True

        self.current_iteration += 1
        return False
