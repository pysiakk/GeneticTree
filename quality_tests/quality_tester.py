
from genetic_tree import GeneticTree
import pandas as pd
import numpy as np
from genetic.initializer import InitializationType
from genetic.selector import SelectionType
from genetic.evaluator import Metric
import json


def test_over_params(X_train: list, y_train: list, X_test: list, y_test: list, dataset: str,
                     iterate_over_1: str, iterate_params_1: list, json_path,
                     n_trees: int = 400,
                     n_thresholds: int = 10,
                     initial_depth: int = 1,
                     initialization_type: InitializationType = InitializationType.Random,
                     split_prob: float = 0.7,
                     mutation_prob: float = 0.4,
                     mutations_additional: list = None,
                     mutation_is_replace: bool = False,
                     cross_prob: float = 0.6,
                     cross_is_both: bool = True,
                     is_left_selected_parents: bool = False,
                     max_iterations: int = 500,
                     selection_type: SelectionType = SelectionType.StochasticUniform,
                     n_elitism: int = 3,
                     metric: Metric = Metric.AccuracyMinusDepth,
                     remove_other_trees: bool = True,
                     remove_variables: bool = True,
                     seed: int = None,
                     save_metrics: bool = True,
                     verbose: bool = True,
                     n_jobs: int = -1):

    test_records = []
    for iter_1 in iterate_params_1:
        parms = {
            "n_trees": int(n_trees),
            "n_thresholds": int(n_thresholds),
            "initial_depth": int(initial_depth),
            "initialization_type": initialization_type.name,
            "split_prob": float(split_prob),
            "mutation_prob": float(mutation_prob),
            "mutations_additional": mutations_additional,
            "mutation_is_replace": mutation_is_replace,
            "cross_prob": float(cross_prob),
            "cross_is_both": cross_is_both,
            "is_left_selected_parents": is_left_selected_parents,
            "max_iterations": int(max_iterations),
            "selection_type": selection_type.name,
            "n_elitism": int(n_elitism),
            "metric": metric.name,
            "remove_other_trees": remove_other_trees,
            "remove_variables": remove_variables,
            "seed": seed,
            "save_metrics": save_metrics,
            "verbose": verbose,
            "n_jobs": int(n_jobs),
            iterate_over_1: iter_1
        }

        kwargs = {"n_trees": int(n_trees), "n_thresholds": int(n_thresholds), "initial_depth": int(initial_depth),
                  "initialization_type": initialization_type, "split_prob": float(split_prob),
                  "mutation_prob": float(mutation_prob), "mutations_additional": mutations_additional,
                  "mutation_is_replace": mutation_is_replace, "cross_prob": float(cross_prob),
                  "cross_is_both": cross_is_both, "is_left_selected_parents": is_left_selected_parents,
                  "max_iterations": int(max_iterations), "selection_type": selection_type, "n_elitism": int(n_elitism),
                  "metric": metric, "remove_other_trees": remove_other_trees, "remove_variables": remove_variables,
                  "seed": seed, "save_metrics": save_metrics, "verbose": verbose, "n_jobs": int(n_jobs),
                  iterate_over_1: iter_1}

        dataset_records = []
        for X_train_i, y_train_i, X_test_i, y_test_i, dataset_i in zip(X_train, y_train, X_test, y_test, dataset):
            gt = GeneticTree(
                # n_trees=n_trees,
                # n_thresholds=n_thresholds,
                # initial_depth=initial_depth,
                # initialization_type=initialization_type,
                # split_prob=split_prob,
                # mutation_prob=mutation_prob,
                # mutations_additional=mutations_additional,
                # mutation_is_replace=mutation_is_replace,
                # cross_prob=cross_prob,
                # cross_is_both=cross_is_both,
                # is_left_selected_parents=is_left_selected_parents,
                # max_iterations=max_iterations,
                # selection_type=selection_type,
                # n_elitism=n_elitism,
                # metric=metric,
                # remove_other_trees=remove_other_trees,
                # remove_variables=remove_variables,
                # seed=seed,
                # save_metrics=save_metrics,
                # verbose=verbose,
                # n_jobs=n_jobs)
                **kwargs)
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
    data = pd.read_csv("~/Desktop/diabetes.csv")
    y_train = np.array(data.Outcome)[0:700]
    X_train = np.array(data.iloc[:700, :8])
    y_test = np.array(data.Outcome)[700:]
    X_test = np.array(data.iloc[700:, :8])
    data = test_over_params([X_train], [y_train], [X_test], [y_test], ["diabetes"], "max_iterations", [1, 2, 3], "quality_tests/test_json_1.json")
    print(data)
    data = test_over_params([X_train], [y_train], [X_test], [y_test], ["diabetes"], "cross_prob", [0.5, 0.8], "quality_tests/test_json_2.json",
                            max_iterations=5)
    print(data)
