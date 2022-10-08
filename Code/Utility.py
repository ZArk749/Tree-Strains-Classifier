import re
import json
import os
from sklearn.model_selection import GridSearchCV
from Code.TreeStrains import TreeStrainsClassifier
from sklearn.metrics import pairwise_kernels
import numpy as np


def write_results(dataset_key, key, value):
    path = "./Results/JSON/{}.json".format(dataset_key)
    if not os.path.exists(path):
        dictionary = {}
        with open(path, 'w') as outfile:s
            json.dump(dictionary, outfile, indent=4)

    if os.path.exists(path):
        json_file = open(path, "r")
        dictionary = json.load(json_file)
        json_file.close()

        dictionary[key] = value

        with open(path, 'w') as outfile:
            json.dump(dictionary, outfile, indent=4)


def extract_dataset_name(url):
    matches = re.finditer('/', url)
    matches_positions = [match.start() for match in matches]

    lastSlash = matches_positions[-1] + 1

    return url[lastSlash:-4]


def find_best_metric(X, y, dataset):
    metrics = ['additive_chi2', 'chi2', 'linear', 'poly', 'polynomial', 'rbf', 'laplacian', 'sigmoid', 'cosine']
    valid_metrics = []
    for metric in metrics:
        try:
            pairwise_kernels(X, metric=metric, n_jobs=-1)
            valid_metrics.append(metric)
        except ValueError as ve:
            pass

    param_grid = [
        {'metric': valid_metrics,
         'n_jobs': [-1],
         'autofill': [True]}]

    classifier = TreeStrainsClassifier()

    grid_search = GridSearchCV(classifier, param_grid, cv=3,  # three folds crossVal
                               scoring='neg_mean_squared_error',
                               return_train_score=True)
    grid_search.fit(X, y)

    best_params = grid_search.best_params_['metric']

    del classifier
    del grid_search

    params = {extract_dataset_name(dataset): best_params}

    path = "./Datasets/.best_params.json"

    if not os.path.exists(path):
        with open(path, 'w') as outfile:
            json.dump(params, outfile, indent=4)
    if os.path.exists(path):
        json_file = open(path, "r")
        dictionary = json.load(json_file)
        json_file.close()

        dictionary[extract_dataset_name(dataset)] = best_params

        with open(path, 'w') as outfile:
            json.dump(dictionary, outfile, indent=4)
            

def load_dict(path):
    json_file = open(path, "r")
    dictionary = json.load(json_file)
    json_file.close()
    return dictionary
