from genetic_tree.genetic_tree import GeneticTree
import pandas as pd
import numpy as np
from genetic_tree.genetic.initializer import Initialization
from genetic_tree.genetic.selector import Selection
from genetic_tree.genetic.evaluator import Metric
import json
import mnist
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import openml
from pytictoc import TicToc
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def over_parms_test(X_train: list, y_train: list, X_test: list, y_test: list, dataset: list,
                     iterate_over_1: str, iterate_params_1: list, json_path,
                     n_trees: int = 400,
                     max_iter: int = 500,
                     cross_prob: float = 0.6,
                     mutation_prob: float = 0.4,
                     initialization: Initialization = Initialization.Full,
                     metric: Metric = Metric.AccuracyMinusDepth,
                     selection: Selection = Selection.StochasticUniform,
                     n_elitism: int = 3,
                     n_thresholds: int = 10,
                     cross_both: bool = True,
                     mutations_additional: list = None,
                     mutation_replace: bool = False,
                     initial_depth: int = 1,
                     split_prob: float = 0.7,
                     n_leaves_factor: float = 0.0001,
                     depth_factor: float = 0.001,
                     tournament_size: int = 3,
                     leave_selected_parents: bool = False,
                     n_iter_no_change: int = 100,
                     early_stopping: bool = False,
                     random_state: int = 123,
                     save_metrics: bool = True,
                     keep_last_population: bool = False,
                     remove_variables: bool = True,
                     verbose: bool = True,
                     n_jobs: int = -1):
    test_records = []
    for iter_1 in iterate_params_1:
        parms = {
            "n_trees": int(n_trees),
            "max_iter": int(max_iter),
            "cross_prob": float(cross_prob),
            "mutation_prob": float(mutation_prob),
            "initialization": initialization.name,
            "metric": metric.name,
            "selection": selection.name,
            "n_elitism": int(n_elitism),
            "n_thresholds": int(n_thresholds),
            "cross_both": cross_both,
            "mutations_additional": mutations_additional,
            "mutation_replace": mutation_replace,
            "initial_depth": int(initial_depth),
            "split_prob": float(split_prob),
            "n_leaves_factor": float(n_leaves_factor),
            "depth_factor": float(depth_factor),
            "tournament_size": int(tournament_size),
            "leave_selected_parents": leave_selected_parents,
            "n_iter_no_change": int(n_iter_no_change),
            "early_stopping": early_stopping,
            "random_state": random_state,
            "save_metrics": save_metrics,
            "keep_last_population": keep_last_population,
            "remove_variables": remove_variables,
            "verbose": verbose,
            "n_jobs": int(n_jobs)
        }
        if iterate_over_1 in ["initialization", "selection", "metric"]:
            parms[iterate_over_1] = iter_1.name
        else:
            parms[iterate_over_1] = iter_1

        kwargs = {"n_trees": int(n_trees),
                  "max_iter": int(max_iter),
                  "cross_prob": float(cross_prob),
                  "mutation_prob": float(mutation_prob),
                  "initialization": initialization,
                  "metric": metric,
                  "selection": selection,
                  "n_elitism": int(n_elitism),
                  "n_thresholds": int(n_thresholds),
                  "cross_both": cross_both,
                  "mutations_additional": mutations_additional,
                  "mutation_replace": mutation_replace,
                  "initial_depth": int(initial_depth),
                  "split_prob": float(split_prob),
                  "n_leaves_factor": float(n_leaves_factor),
                  "depth_factor": float(depth_factor),
                  "tournament_size": int(tournament_size),
                  "leave_selected_parents": leave_selected_parents,
                  "n_iter_no_change": int(n_iter_no_change),
                  "early_stopping": early_stopping,
                  "random_state": random_state,
                  "save_metrics": save_metrics,
                  "keep_last_population": keep_last_population,
                  "remove_variables": remove_variables,
                  "verbose": verbose,
                  "n_jobs": int(n_jobs),
                  iterate_over_1: iter_1}

        dataset_records = []
        for X_train_i, y_train_i, X_test_i, y_test_i, dataset_i in zip(X_train, y_train, X_test, y_test, dataset):
            print(dataset_i + ":")
            gt = GeneticTree(**kwargs)
            t = TicToc()
            t.tic()
            gt.fit(X=X_train_i, y=y_train_i)
            t.toc()
            print("GT_Correct: " + str(sum(gt.predict(X_test_i) == y_test_i)))
            print("GT_All: " + str(len(y_test_i)))
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

    diabetes = pd.read_csv("https://www.openml.org/data/get_csv/37/dataset_37_diabetes.csv").sample(frac=1,
                                                                                                    random_state=123)
    print("diabetes: " + str(diabetes.shape))
    diabetes["class"] = LabelEncoder().fit_transform(diabetes["class"])
    diabetes_X_train = np.array(diabetes.iloc[diabetes.shape[0] // 7:, :diabetes.shape[1] - 1])
    diabetes_y_train = np.array(diabetes["class"])[diabetes.shape[0] // 7:]
    diabetes_X_test = np.array(diabetes.iloc[:diabetes.shape[0] // 7, :diabetes.shape[1] - 1])
    diabetes_y_test = np.array(diabetes["class"])[:diabetes.shape[0] // 7]

    print("diabetes: " + str(len(diabetes_y_train)) + ", " + str(len(diabetes_y_test)))

    ozone = pd.read_csv("https://www.openml.org/data/get_csv/1592279/phpdReP6S.csv").sample(frac=1,
                                                                                            random_state=123)
    print("ozone: " + str(ozone.shape))
    ozone_X_train = np.array(ozone.iloc[ozone.shape[0] // 7:, :ozone.shape[1] - 1])
    ozone_y_train = np.array(ozone["Class"])[ozone.shape[0] // 7:] - 1
    ozone_X_test = np.array(ozone.iloc[:ozone.shape[0] // 7, :ozone.shape[1] - 1])
    ozone_y_test = np.array(ozone["Class"])[:ozone.shape[0] // 7] - 1

    print("ozone: " + str(len(ozone_y_train)) + ", " + str(len(ozone_y_test)))

    banknote = pd.read_csv("https://www.openml.org/data/get_csv/1586223/php50jXam.csv").sample(frac=1,
                                                                                               random_state=123)
    print("banknote: " + str(banknote.shape))
    banknote_X_train = np.array(banknote.iloc[banknote.shape[0] // 7:, :banknote.shape[1] - 1])
    banknote_y_train = np.array(banknote["Class"])[banknote.shape[0] // 7:] - 1
    banknote_X_test = np.array(banknote.iloc[:banknote.shape[0] // 7, :banknote.shape[1] - 1])
    banknote_y_test = np.array(banknote["Class"])[:banknote.shape[0] // 7] - 1

    print("banknote: " + str(len(banknote_y_train)) + ", " + str(len(banknote_y_test)))

    plants = pd.read_csv("https://www.openml.org/data/get_csv/1592285/phpoOxxNn.csv").sample(frac=1,
                                                                                             random_state=123)
    print("plants: " + str(plants.shape))
    plants["Class"] = LabelEncoder().fit_transform(plants["Class"])
    # print(plants)
    plants_X_train = np.array(plants.iloc[plants.shape[0] // 7:, :plants.shape[1] - 1])
    plants_y_train = np.array(plants["Class"])[plants.shape[0] // 7:] - 1
    plants_X_test = np.array(plants.iloc[:plants.shape[0] // 7, :plants.shape[1] - 1])
    plants_y_test = np.array(plants["Class"])[:plants.shape[0] // 7] - 1

    print("plants: " + str(len(plants_y_train)) + ", " + str(len(plants_y_test)))

    madelon = pd.read_csv("https://www.openml.org/data/get_csv/1590986/phpfLuQE4.csv").sample(frac=1,
                                                                                             random_state=123)
    print("madelon: " + str(madelon.shape))
    madelon["Class"] = LabelEncoder().fit_transform(madelon["Class"])
    madelon_X_train = np.array(madelon.iloc[madelon.shape[0] // 7:, :madelon.shape[1] - 1])
    madelon_y_train = np.array(madelon["Class"])[madelon.shape[0] // 7:] - 1
    madelon_X_test = np.array(madelon.iloc[:madelon.shape[0] // 7, :madelon.shape[1] - 1])
    madelon_y_test = np.array(madelon["Class"])[:madelon.shape[0] // 7] - 1

    print("madelon: " + str(len(madelon_y_train)) + ", " + str(len(madelon_y_test)))

    abalone = pd.read_csv("https://www.openml.org/data/get_csv/3620/dataset_187_abalone.csv").sample(frac=1,
                                                                                                     random_state=123)
    print("abalone: " + str(abalone.shape))
    abalone_X = abalone.iloc[:, :abalone.shape[1] - 1]
    abalone_y = abalone["Class_number_of_rings"]
    abalone_X = OneHotEncoder().fit_transform(abalone_X).toarray()
    abalone_X_train = np.array(abalone_X[abalone_X.shape[0] // 7:, :])
    abalone_y_train = np.array(abalone_y)[abalone_y.shape[0] // 7:] - 1
    abalone_X_test = np.array(abalone_X[:abalone_X.shape[0] // 7, :])
    abalone_y_test = np.array(abalone_y)[:abalone_y.shape[0] // 7] - 1
    print(abalone_X)

    print("abalone: " + str(len(abalone_y_train)) + ", " + str(len(abalone_y_test)))

    print("mnist: (70000, 785)")
    mnist_X_train = mnist.train_images().reshape((60000, 784))
    mnist_y_train = mnist.train_labels()
    mnist_X_test = mnist.test_images().reshape((10000, 784))
    mnist_y_test = mnist.test_labels()

    print("mnist: " + str(60000) + ", " + str(10000))

    train_X_list = [diabetes_X_train, ozone_X_train, banknote_X_train, plants_X_train,
                    madelon_X_train, abalone_X_train,
                    mnist_X_train]
    train_y_list = [diabetes_y_train, ozone_y_train, banknote_y_train, plants_y_train,
                    madelon_y_train, abalone_y_train,
                    mnist_y_train]
    test_X_list = [diabetes_X_test, ozone_X_test, banknote_X_test, plants_X_test,
                   madelon_X_test, abalone_X_test,
                   mnist_X_test]
    test_y_list = [diabetes_y_test, ozone_y_test, banknote_y_test, plants_y_test,
                   madelon_y_test, abalone_y_test,
                   mnist_y_test]
    dataset_list = ["diabetes", "ozone", "banknote", "plants",
                     "madelon", "abalone", "mnist"]

    
    cross_prob = over_parms_test(train_X_list, train_y_list, test_X_list, test_y_list, dataset_list,
                                  "cross_prob", [0.2, 0.4, 0.6, 0.8, 1],
                                  "quality_tests/cross_prob_1.json")

    mut_prob = over_parms_test(train_X_list, train_y_list, test_X_list, test_y_list, dataset_list,
                                "mutation_prob", [0.2, 0.4, 0.6, 0.8, 1],
                                "quality_tests/mut_prob_1.json")

    n_trees = over_parms_test(train_X_list, train_y_list, test_X_list, test_y_list, dataset_list,
                               "n_trees", [10, 50, 150, 500, 1000],
                               "quality_tests/n_trees_1.json")

    n_thresholds = over_parms_test(train_X_list, train_y_list, test_X_list, test_y_list, dataset_list,
                                    "n_thresholds", [3, 10, 30, 100, 300],
                                    "quality_tests/n_thresholds_1.json")

    metrics = over_parms_test(train_X_list, train_y_list, test_X_list, test_y_list, dataset_list,
                               "metric", [Metric.Accuracy, Metric.AccuracyMinusDepth, Metric.AccuracyMinusLeavesNumber],
                               "quality_tests/metrics_1.json")

    selection = over_parms_test(train_X_list, train_y_list, test_X_list, test_y_list, dataset_list,
                                 "selection", [Selection.StochasticUniform, Selection.Tournament, Selection.Roulette,
                                               Selection.Rank],
                                 "quality_tests/selection_1.json")

    initialization = over_parms_test(train_X_list, train_y_list, test_X_list, test_y_list, dataset_list,
                                      "initialization",
                                      [Initialization.Full, Initialization.Half, Initialization.Split],
                                      "quality_tests/initialization_1.json",
                                     initial_depth=10)

    tournament_size = over_parms_test(train_X_list, train_y_list, test_X_list, test_y_list, dataset_list,
                                      "tournament_size",
                                      [2, 3, 5, 8, 13],
                                      "quality_tests/tournament_size_1.json")

    n_elitism = over_parms_test(train_X_list, train_y_list, test_X_list, test_y_list, dataset_list,
                                 "n_elitism", [1, 3, 5, 20, 100],
                                 "quality_tests/elitism_1.json")

    n_leaves_factor = over_parms_test(train_X_list, train_y_list, test_X_list, test_y_list, dataset_list,
                                       "n_leaves_factor", [0.000001, 0.00001, 0.0001, 0.001, 0.01],
                                       "quality_tests/n_leaves_factor_1.json",
                                       metric=Metric.AccuracyMinusLeavesNumber)

    depth_factor = over_parms_test(train_X_list, train_y_list, test_X_list, test_y_list, dataset_list,
                                    "depth_factor", [0.00001, 0.0001, 0.001, 0.01, 0.1],
                                    "quality_tests/depth_factor_1.json",
                                    metric=Metric.AccuracyMinusDepth)

    initial_depth = over_parms_test(train_X_list, train_y_list, test_X_list, test_y_list, dataset_list,
                                     "initial_depth", [1, 3, 5, 8, 13],
                                     "quality_tests/initial_depth_1.json",
                                     initialization=Initialization.Split)


    initial_depth = over_parms_test([diabetes_X_train], [diabetes_y_train], [diabetes_X_test], [diabetes_y_test], ["diabetes"],
                                     "initial_depth", [10],
                                     "quality_tests/diabetes_test_4.json",
                                     initialization=Initialization.Split,
                                     metric=Metric.AccuracyMinusLeavesNumber,
                                     n_leaves_factor=0.0000001,
                                     random_state=123,
                                     selection=Selection.Tournament,
                                     n_elitism=3,
                                     max_iter=2000)
    
    dt = DecisionTreeClassifier(random_state=123)
    t = TicToc()
    t.tic()
    dt.fit(diabetes_X_train, diabetes_y_train)
    t.toc()
    print(dt.get_n_leaves())
    y_pred = list(dt.predict(diabetes_X_test))
    print("DT_Correct: " + str(sum(y_pred == diabetes_y_test)))
    print("DT_All: " + str(len(diabetes_y_test)))
    print(sum(y_pred == diabetes_y_test)/len(y_pred))
    
    rf = RandomForestClassifier(random_state=123)
    t = TicToc()
    t.tic()
    rf.fit(diabetes_X_train, diabetes_y_train)
    t.toc()
    y_pred = list(rf.predict(diabetes_X_test))
    print("RF_Correct: " + str(sum(y_pred == diabetes_y_test)))
    print("RF_All: " + str(len(diabetes_y_test)))
    print(sum(y_pred==diabetes_y_test)/len(y_pred))

    initial_depth = over_parms_test([ozone_X_train], [ozone_y_train], [ozone_X_test], [ozone_y_test], ["ozone"],
                                     "initial_depth", [10],
                                     "quality_tests/ozone_test_4.json",
                                     initialization=Initialization.Split,
                                     metric=Metric.AccuracyMinusLeavesNumber,
                                     n_leaves_factor=0.0000001,
                                     random_state=123,
                                     selection=Selection.Tournament,
                                     n_elitism=3,
                                     max_iter=2000)
    
    dt = DecisionTreeClassifier(random_state=123)
    t = TicToc()
    t.tic()
    dt.fit(ozone_X_train, ozone_y_train)
    t.toc()
    print(dt.get_n_leaves())
    y_pred = list(dt.predict(ozone_X_test))
    print("DT_Correct: " + str(sum(y_pred == ozone_y_test)))
    print("DT_All: " + str(len(ozone_y_test)))
    print(sum(y_pred==ozone_y_test)/len(y_pred))

    rf = RandomForestClassifier(random_state=123)
    t = TicToc()
    t.tic()
    rf.fit(ozone_X_train, ozone_y_train)
    t.toc()
    y_pred = list(rf.predict(ozone_X_test))
    print("RF_Correct: " + str(sum(y_pred == ozone_y_test)))
    print("RF_All: " + str(len(ozone_y_test)))
    print(sum(y_pred==ozone_y_test)/len(y_pred))

    initial_depth = over_parms_test([banknote_X_train], [banknote_y_train], [banknote_X_test], [banknote_y_test], ["banknote"],
                                     "initial_depth", [10],
                                     "quality_tests/banknote_test_4.json",
                                     initialization=Initialization.Split,
                                     metric=Metric.AccuracyMinusLeavesNumber,
                                     n_leaves_factor=0.0000001,
                                     random_state=123,
                                     selection=Selection.Tournament,
                                     n_elitism=3,
                                     max_iter=2000)
    
    dt = DecisionTreeClassifier(random_state=123)
    t = TicToc()
    t.tic()
    dt.fit(banknote_X_train, banknote_y_train)
    t.toc()
    print(dt.get_n_leaves())
    y_pred = list(dt.predict(banknote_X_test))
    print("DT_Correct: " + str(sum(y_pred == banknote_y_test)))
    print("DT_All: " + str(len(banknote_y_test)))
    print(sum(y_pred==banknote_y_test)/len(y_pred))
    
    rf = RandomForestClassifier(random_state=123)
    t = TicToc()
    t.tic()
    rf.fit(banknote_X_train, banknote_y_train)
    t.toc()
    y_pred = list(rf.predict(banknote_X_test))
    print("RF_Correct: " + str(sum(y_pred == banknote_y_test)))
    print("RF_All: " + str(len(banknote_y_test)))
    print(sum(y_pred==banknote_y_test)/len(y_pred))

    initial_depth = over_parms_test([plants_X_train], [plants_y_train], [plants_X_test], [plants_y_test], ["plants"],
                                     "initial_depth", [10],
                                     "quality_tests/plants_test_1.json",
                                     initialization=Initialization.Split,
                                     metric=Metric.AccuracyMinusLeavesNumber,
                                     n_leaves_factor=0.00000001,
                                     random_state=123,
                                     n_thresholds=50,
                                     selection=Selection.Tournament,
                                     n_elitism=3,
                                     mutation_prob=0.8,
                                     cross_prob=0.8,
                                     max_iter=2000)
    
    dt = DecisionTreeClassifier(random_state=123)
    t = TicToc()
    t.tic()
    dt.fit(plants_X_train, plants_y_train)
    t.toc()
    print(dt.get_n_leaves())
    y_pred = list(dt.predict(plants_X_test))
    print("DT_Correct: " + str(sum(y_pred == plants_y_test)))
    print("DT_All: " + str(len(plants_y_test)))
    print(sum(y_pred==plants_y_test)/len(y_pred))
    
    rf = RandomForestClassifier(random_state=123)
    t = TicToc()
    t.tic()
    rf.fit(plants_X_train, plants_y_train)
    t.toc()
    y_pred = list(rf.predict(plants_X_test))
    print("RF_Correct: " + str(sum(y_pred == plants_y_test)))
    print("RF_All: " + str(len(plants_y_test)))
    print(sum(y_pred==plants_y_test)/len(y_pred))

    initial_depth = over_parms_test([madelon_X_train], [madelon_y_train], [madelon_X_test], [madelon_y_test], ["madelon"],
                                     "initial_depth", [10],
                                     "quality_tests/madelon_test_4.json",
                                     initialization=Initialization.Split,
                                     metric=Metric.AccuracyMinusLeavesNumber,
                                     n_leaves_factor=0.0000001,
                                     random_state=123,
                                     selection=Selection.Tournament,
                                     n_elitism=3,
                                     max_iter=2000)
    
    dt = DecisionTreeClassifier(random_state=123)
    t = TicToc()
    t.tic()
    dt.fit(madelon_X_train, madelon_y_train)
    t.toc()
    print(dt.get_n_leaves())
    y_pred = list(dt.predict(madelon_X_test))
    print("DT_Correct: " + str(sum(y_pred == madelon_y_test)))
    print("DT_All: " + str(len(madelon_y_test)))
    print(sum(y_pred==madelon_y_test)/len(y_pred))
    
    rf = RandomForestClassifier(random_state=123)
    t = TicToc()
    t.tic()
    rf.fit(madelon_X_train, madelon_y_train)
    t.toc()
    y_pred = list(rf.predict(madelon_X_test))
    print("RF_Correct: " + str(sum(y_pred == madelon_y_test)))
    print("RF_All: " + str(len(madelon_y_test)))
    print(sum(y_pred==madelon_y_test)/len(y_pred))

    initial_depth = over_parms_test([abalone_X_train], [abalone_y_train], [abalone_X_test], [abalone_y_test], ["abalone"],
                                     "initial_depth", [10],
                                     "quality_tests/abalone_test_4.json",
                                     initialization=Initialization.Split,
                                     metric=Metric.AccuracyMinusLeavesNumber,
                                     n_leaves_factor=0.0000001,
                                     random_state=123,
                                     selection=Selection.Tournament,
                                     n_elitism=3,
                                     max_iter=2000)

    dt = DecisionTreeClassifier(random_state=123)
    t = TicToc()
    t.tic()
    dt.fit(abalone_X_train, abalone_y_train)
    print(dt.get_n_leaves())
    t.toc()
    y_pred = list(dt.predict(abalone_X_test))
    print("DT_Correct: " + str(sum(y_pred == abalone_y_test)))
    print("DT_All: " + str(len(abalone_y_test)))
    print(sum(y_pred==abalone_y_test)/len(y_pred))

    rf = RandomForestClassifier(random_state=123)
    t = TicToc()
    t.tic()
    rf.fit(abalone_X_train, abalone_y_train)
    t.toc()
    y_pred = list(rf.predict(abalone_X_test))
    print("RF_Correct: " + str(sum(y_pred == abalone_y_test)))
    print("RF_All: " + str(len(abalone_y_test)))
    print(sum(y_pred==abalone_y_test)/len(y_pred))

    initial_depth = over_parms_test([mnist_X_train], [mnist_y_train], [mnist_X_test], [mnist_y_test], ["mnist"],
                                     "initial_depth", [10],
                                     "quality_tests/mnist_test_4.json",
                                     initialization=Initialization.Split,
                                     metric=Metric.AccuracyMinusLeavesNumber,
                                     n_leaves_factor=0.0000001,
                                     random_state=123,
                                     selection=Selection.Tournament,
                                     n_elitism=3,
                                     max_iter=2000)

    dt = DecisionTreeClassifier(random_state=123)
    t = TicToc()
    t.tic()
    dt.fit(mnist_X_train, mnist_y_train)
    print(dt.get_n_leaves())
    t.toc()
    y_pred = list(dt.predict(mnist_X_test))
    print("DT_Correct: " + str(sum(y_pred == mnist_y_test)))
    print("DT_All: " + str(len(mnist_y_test)))
    print(sum(y_pred == mnist_y_test)/len(y_pred))

    rf = RandomForestClassifier(random_state=123)
    t = TicToc()
    t.tic()
    rf.fit(mnist_X_train, mnist_y_train)
    t.toc()
    y_pred = list(rf.predict(mnist_y_test))
    print("RF_Correct: " + str(sum(y_pred == mnist_y_test)))
    print("RF_All: " + str(len(mnist_y_test)))
    print(sum(y_pred == mnist_y_test)/len(y_pred))
