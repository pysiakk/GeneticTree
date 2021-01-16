import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



def fig_to_file(name):
    file = open("quality_tests/results/" + name + ".json", "r")
    data = json.load(file)

    leaves = []
    depth = []
    acc = []
    keys = []
    gen = []

    iter_over = data['iter_over']
    print(data['test_records'][0]['parms'][iter_over])

    for i in range(len(data['test_records'])):
        leaves_i = np.zeros(len(data['test_records'][i]['dataset_records'][0]['n_leaves_mean']))
        depth_i = np.zeros(len(data['test_records'][i]['dataset_records'][0]['n_leaves_mean']))
        acc_i = np.zeros(len(data['test_records'][i]['dataset_records'][0]['n_leaves_mean']))
        for j in range(len(data['test_records'][i]['dataset_records'])):
            leaves_i = np.add(leaves_i, data['test_records'][i]['dataset_records'][j]['n_leaves_mean'])
            depth_i = np.add(depth_i, data['test_records'][i]['dataset_records'][j]['depth_mean'])
            acc_i = np.add(acc_i, data['test_records'][i]['dataset_records'][j]['acc_best'])
        leaves = leaves + list(leaves_i/len(data['test_records'][i]['dataset_records']))
        depth = depth + list(depth_i/len(data['test_records'][i]['dataset_records']))
        acc = acc + list(acc_i/len(data['test_records'][i]['dataset_records']))
        keys = keys + [data['test_records'][i]['parms'][iter_over]] * len(data['test_records'][i]['dataset_records'][0]['n_leaves_mean'])
        gen = gen + [k + 1 for k in range(len(data['test_records'][i]['dataset_records'][0]['n_leaves_mean']))]

    df = pd.DataFrame({iter_over: keys, "n_leaves_mean": leaves, "depth_mean": depth, "generation": gen, "acc_best": acc})
    df.set_index('generation', inplace=True)
    df.groupby(iter_over)["n_leaves_mean"].plot(logy=True)
    plt.ylabel("mean n_leaves_mean")
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left", title=iter_over)
    plt.savefig("quality_tests/images/" + name + "_n_leaves_mean.png", bbox_inches='tight')
    plt.clf()
    df = pd.DataFrame({iter_over: keys, "n_leaves_mean": leaves, "depth_mean": depth, "generation": gen, "acc_best": acc})
    df.set_index('generation', inplace=True)
    df.groupby(iter_over)["depth_mean"].plot(logy=False)
    plt.ylabel("mean depth_mean")
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left", title=iter_over)
    plt.savefig("quality_tests/images/" + name + "_depth_mean.png", bbox_inches='tight')
    plt.clf()
    df = pd.DataFrame({iter_over: keys, "n_leaves_mean": leaves, "depth_mean": depth, "generation": gen, "acc_best": acc})
    df.set_index('generation', inplace=True)
    df.groupby(iter_over)["acc_best"].plot(logy=False)
    plt.ylabel("mean accuracy_best")
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left", title=iter_over)
    plt.savefig("quality_tests/images/" + name + "_acc_best.png", bbox_inches='tight')
    plt.clf()

names = ["cross_prob_1", "depth_factor_1", "elitism_1", "initial_depth_1", "initialization_1", "metrics_1", "mut_prob_1",
         "n_leaves_factor_1", "n_thresholds_1", "n_trees_1", "selection_1", "tournament_size_1"]

for name in names:
    fig_to_file(name)


acc = pd.DataFrame([
    ["GeneticTree", "diabetes", 0.7431],
    ["DecisionTree", "diabetes", 0.7431],
    ["RandomForest", "diabetes", 0.7706],
    ["GeneticTree", "ozone", 0.9309],
    ["DecisionTree", "ozone", 0.8785],
    ["RandomForest", "ozone", 0.9365],
    ["GeneticTree", "banknote", 0.9898],
    ["DecisionTree", "banknote", 0.9949],
    ["RandomForest", "banknote", 1.0],
    ["GeneticTree", "plants", 0.1622],
    ["DecisionTree", "plants", 0.5263],
    ["RandomForest", "plants", 0.8246],
    ["GeneticTree", "madelon", 0.7763],
    ["DecisionTree", "madelon", 0.7116],
    ["RandomForest", "madelon", 0.6927],
    ["GeneticTree", "abalone", 0.2550],
    ["DecisionTree", "abalone", 0.2181],
    ["RandomForest", "abalone", 0.2232],
    ["GeneticTree", "mnist", 0.7638],
    ["DecisionTree", "mnist", 0.8776],
    ["RandomForest", "mnist", 0.9697]
], columns=['classifier', 'dataset', 'value'])

time = pd.DataFrame([
    ["GeneticTree", "diabetes", 178.928485],
    ["DecisionTree", "diabetes", 0.004499],
    ["RandomForest", "diabetes", 0.144840],
    ["GeneticTree", "ozone", 34.995074],
    ["DecisionTree", "ozone", 0.083739],
    ["RandomForest", "ozone", 0.510958],
    ["GeneticTree", "banknote", 23.756215],
    ["DecisionTree", "banknote", 0.002368],
    ["RandomForest", "banknote", 0.158727],
    ["GeneticTree", "plants", 292.662415],
    ["DecisionTree", "plants", 0.082996],
    ["RandomForest", "plants", 0.787428],
    ["GeneticTree", "madelon", 348.548228],
    ["DecisionTree", "madelon", 0.541019],
    ["RandomForest", "madelon", 1.889679],
    ["GeneticTree", "abalone", 1091.539640],
    ["DecisionTree", "abalone", 8.303404],
    ["RandomForest", "abalone", 16.439596],
    ["GeneticTree", "mnist", 4796.033816],
    ["DecisionTree", "mnist", 15.103409],
    ["RandomForest", "mnist", 29.703812]
], columns=['classifier', 'dataset', 'value'])

n_leaves = pd.DataFrame([
    ["GeneticTree", "diabetes", 71],
    ["DecisionTree", "diabetes", 122],
    ["GeneticTree", "ozone", 1],
    ["DecisionTree", "ozone", 100],
    ["GeneticTree", "banknote", 33],
    ["DecisionTree", "banknote", 28],
    ["GeneticTree", "plants", 72],
    ["DecisionTree", "plants", 415],
    ["GeneticTree", "madelon", 62],
    ["DecisionTree", "madelon", 190],
    ["GeneticTree", "abalone", 79],
    ["DecisionTree", "abalone", 2212],
    ["GeneticTree", "mnist", 138],
    ["DecisionTree", "mnist", 3943]
], columns=['classifier', 'dataset', 'value'])

acc.pivot('dataset', 'classifier', 'value').plot(kind='bar')
plt.legend(bbox_to_anchor=(1, 1), loc="upper left", title="Classifier")
plt.ylabel('accuracy')
plt.show()
#
time.pivot('dataset', 'classifier', 'value').plot(kind='bar', logy=True)
plt.legend(bbox_to_anchor=(1, 1), loc="upper left", title="Classifier")
plt.ylabel('time[s]')
plt.show()

n_leaves.pivot('dataset', 'classifier', 'value').plot(kind='bar', logy=True)
plt.legend(bbox_to_anchor=(1, 1), loc="upper left", title="Classifier")
plt.ylabel('n_leaves')
plt.show()
