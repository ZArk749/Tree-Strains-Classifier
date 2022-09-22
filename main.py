from Code.CV import CrossValidation
import settings as clf
import pandas as pd
import numpy as np
import re
import json
import os


def WriteOnDict(keyDataset, Key, Value):
    path = "./Results/JSON/{}.json".format(keyDataset)
    if not os.path.exists(path):
        dictionary = {}
        with open(path, 'w') as outfile:
            json.dump(dictionary, outfile, indent=4)

    if os.path.exists(path):
        json_file = open(path, "r")
        dictionary = json.load(json_file)
        json_file.close()

        dictionary[Key] = Value

        with open(path, 'w') as outfile:
            json.dump(dictionary, outfile, indent=4)


def ExtractNameDataset(url):
    matches = re.finditer('/', url)
    matches_positions = [match.start() for match in matches]

    lastSlash = matches_positions[-1] + 1

    return url[lastSlash:-4]

if __name__ == '__main__':

    datasets = clf.getDataset()

    model = clf.getClassifier()

    for i in datasets:
        print(i)
        dataset = pd.read_csv(i, header=0)

        X = dataset.iloc[:, :len(dataset.columns) - 1].to_numpy()
        y = dataset.iloc[:, -1:].to_numpy()
        y = np.ravel(y.astype(int))

        metrics = CrossValidation(model, X, y)
        # print(metrics)

        nameDataset = ExtractNameDataset(i)

        for key in metrics:
            WriteOnDict(nameDataset, 'TreeStrains' + key, metrics[key])





