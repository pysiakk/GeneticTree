
from genetic_tree import GeneticTree
import pandas as pd
import numpy as np
from genetic.initializer import Initialization
from genetic.selector import Selection
from genetic.evaluator import Metric
import json


def test_over_params(X_train: list, y_train: list, X_test: list, y_test: list, dataset: list,
                     iterate_over_1: str, iterate_params_1: list, json_path,
                     n_trees: int = 400,
                     n_iters: int = 500,
                     cross_prob: float = 0.6,
                     mutation_prob: float = 0.4,
                     initialization: Initialization = Initialization.Full,
                     metric: Metric = Metric.AccuracyMinusDepth,
                     selection: Selection = Selection.StochasticUniform,
                     n_elitism: int = 3,
                     n_thresholds: int = 10,
                     cross_is_both: bool = True,
                     mutations_additional: list = None,
                     mutation_is_replace: bool = False,
                     initial_depth: int = 1,
                     split_prob: float = 0.7,
                     n_leaves_factor: float = 0.0001,
                     depth_factor: float = 0.01,
                     tournament_size: int = 3,
                     is_leave_selected_parents: bool = False,
                     n_iters_without_improvement: int = 100,
                     use_without_improvement: bool = False,
                     random_state: int = None,
                     is_save_metrics: bool = True,
                     is_keep_last_population: bool = False,
                     is_remove_variables: bool = True,
                     verbose: bool = True,
                     n_jobs: int = -1):

    test_records = []
    for iter_1 in iterate_params_1:
        parms = {
            "n_trees": int(n_trees),
            "n_iters": int(n_iters),
            "cross_prob": float(cross_prob),
            "mutation_prob": float(mutation_prob),
            "initialization": initialization.name,
            "metric": metric.name,
            "selection": selection.name,
            "n_elitism": int(n_elitism),
            "n_thresholds": int(n_thresholds),
            "cross_is_both": cross_is_both,
            "mutations_additional": mutations_additional,
            "mutation_is_replace": mutation_is_replace,
            "initial_depth": int(initial_depth),
            "split_prob": float(split_prob),
            "n_leaves_factor": float(n_leaves_factor),
            "depth_factor": float(depth_factor),
            "tournament_size": int(tournament_size),
            "is_leave_selected_parents": is_leave_selected_parents,
            "n_iters_without_improvement": int(n_iters_without_improvement),
            "use_without_improvement": use_without_improvement,
            "random_state": random_state,
            "is_save_metrics": is_save_metrics,
            "is_keep_last_population": is_keep_last_population,
            "is_remove_variables": is_remove_variables,
            "verbose": verbose,
            "n_jobs": int(n_jobs)
        }
        if iterate_over_1 in ["initialization", "selection", "metric"]:
            parms[iterate_over_1] = iter_1.name
        else:
            parms[iterate_over_1] = iter_1

        kwargs = {"n_trees": int(n_trees),
                  "n_iters": int(n_iters),
                  "cross_prob": float(cross_prob),
                  "mutation_prob": float(mutation_prob),
                  "initialization": initialization,
                  "metric": metric,
                  "selection": selection,
                  "n_elitism": int(n_elitism),
                  "n_thresholds": int(n_thresholds),
                  "cross_is_both": cross_is_both,
                  "mutations_additional": mutations_additional,
                  "mutation_is_replace": mutation_is_replace,
                  "initial_depth": int(initial_depth),
                  "split_prob": float(split_prob),
                  "n_leaves_factor": float(n_leaves_factor),
                  "depth_factor": float(depth_factor),
                  "tournament_size": int(tournament_size),
                  "is_leave_selected_parents": is_leave_selected_parents,
                  "n_iters_without_improvement": int(n_iters_without_improvement),
                  "use_without_improvement": use_without_improvement,
                  "random_state": random_state,
                  "is_save_metrics": is_save_metrics,
                  "is_keep_last_population": is_keep_last_population,
                  "is_remove_variables": is_remove_variables,
                  "verbose": verbose,
                  "n_jobs": int(n_jobs),
                  iterate_over_1: iter_1}

        dataset_records = []
        for X_train_i, y_train_i, X_test_i, y_test_i, dataset_i in zip(X_train, y_train, X_test, y_test, dataset):
            gt = GeneticTree(**kwargs)
            gt.fit(X=X_train_i, y=y_train_i)
            print(sum(gt.predict(X_test_i) == y_test_i) / len(y_test_i))

            dataset_record = {
                "dataset": dataset_i,
                "acc_best": [float(k) for k in gt.acc_best],
                "acc_mean": [float(k) for k in gt.acc_mean],
                "depth_best": [float(k) for k in gt.depth_best],
                "depth_mean": [float(k) for k in gt.depth_mean],
                "n_leaves_best": [float(k) for k in gt.n_leaves_best],
                "n_leaves_mean": [float(k) for k in gt.n_leaves_mean]
            }
            dataset_records.append(dataset_record)

        test_record = {
            "parms": parms,
            "dataset_records": dataset_records
        }
        test_records.append(test_record)

    json_out = {
        "iter_over": iterate_over_1,
        "test_records": test_records
    }
    out_file = open(json_path, "w")
    json.dump(json_out, out_file, indent=4)
    out_file.close()
    return json_out


if __name__ == "__main__":
    diabetes = pd.read_csv("~/Desktop/diabetes.csv")
    diabetes_y_train = np.array(diabetes.Outcome)[0:700]
    diabetes_X_train = np.array(diabetes.iloc[:700, :8])
    diabetes_y_test = np.array(diabetes.Outcome)[700:]
    diabetes_X_test = np.array(diabetes.iloc[700:, :8])
    # data = test_over_params([X_train], [y_train], [X_test], [y_test], ["diabetes"], "max_iterations", [1, 2, 3], "quality_tests/test_json_1.json")
    #
    # data = test_over_params([X_train], [y_train], [X_test], [y_test], ["diabetes"], "cross_prob", [0.5, 0.8], "quality_tests/test_json_2.json",
    #                         max_iterations=5)

    ozone = pd.read_csv("~/Desktop/ozone-level-8hr.csv")
    ozone_y_train = np.array(ozone.Class)[0:2000] - 1
    ozone_X_train = np.array(ozone.iloc[:2000, :72])
    ozone_y_test = np.array(ozone.Class)[2000:] - 1
    ozone_X_test = np.array(ozone.iloc[2000:, :72])

    data = test_over_params([diabetes_X_train, ozone_X_train],
                            [diabetes_y_train, ozone_y_train],
                            [diabetes_X_test, ozone_X_test],
                            [diabetes_y_test, ozone_y_test],
                            ["diabetes", "ozone"],
                            "selection", [Selection.StochasticUniform, Selection.Rank,
                                               Selection.Roulette, Selection.Tournament],
                            "quality_tests/selection_test_1.json",
                            initial_depth=4)

    data = test_over_params([diabetes_X_train, ozone_X_train],
                            [diabetes_y_train, ozone_y_train],
                            [diabetes_X_test, ozone_X_test],
                            [diabetes_y_test, ozone_y_test],
                            ["diabetes", "ozone"],
                            "initialization", [Initialization.Full, Initialization.Half,
                                                    Initialization.Split],
                            "quality_tests/initialization_test_1.json",
                            initial_depth=4)

