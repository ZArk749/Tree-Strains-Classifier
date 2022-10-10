from Code.CV import cross_validation
import settings as clf
import pandas as pd
import numpy as np
from Code import Utility as ut
from Code.SimilarityFunction import SimilarityFunction

from Code.TreeStrains import TreeStrainsClassifier


if __name__ == '__main__':

    datasets = clf.get_datasets()

    for i in datasets:
        print(i)
        dataset = pd.read_csv(i, header=0)

        X = dataset.iloc[:, :len(dataset.columns) - 1].to_numpy()
        y = dataset.iloc[:, -1:].to_numpy()
        y = np.ravel(y.astype(int))

        if clf.do_metric_search():

            ut.find_best_metric(X, y, i)

        else:

            dictionary = ut.load_dict("./Datasets/.best_params.json")

            nameDataset = ut.extract_dataset_name(i)

            metric = dictionary[nameDataset]

            #K = SimilarityFunction(X, metric=metric, n_jobs=-1)

            if clf.do_grid_search():
                model = ut.grid_search(X, y, {'metric': [metric],
                                              'n_jobs': [-1],
                                              'autofill': [True],
                                              'verbose': [0],
                                              'ratio': [1, 0.7, 0.5]})
            else:
                model = TreeStrainsClassifier(metric=metric, ratio=0.7, n_jobs=1, autofill=True)

            print(model)

            metrics = cross_validation(model, X, y)
            # print(metrics)

            for key in metrics:
                ut.write_results(nameDataset, 'TreeStrains' + key, metrics[key])





