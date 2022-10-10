import random

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from random import randint

import numpy as np
import datetime
import warnings

from Code.SimilarityFunction import SimilarityFunction


"""
Validation functions
"""


def _validate_parameters(metric):
    if metric is not None:
        if metric not in ('additive_chi2', 'chi2', 'linear', 'poly', 'polynomial', 'rbf', 'laplacian', 'sigmoid',
                          'cosine', "custom"):
            raise ValueError(
                'Invalid preset "%s" for kernel metric'
                % metric
            )


def _validate_bootstrap(y_sets, n_classes, verbose):

    if verbose > 1:
        index = 1

    for i in y_sets:  # for each set
        if verbose > 1:
            index += 1
        if len(np.unique(i)) != n_classes:
            if verbose > 1:
                print("missing class on set no.", index)
            return False
    return True


"""
Support functions
"""


def _similarity_bootstrap(X, y, K, n_estimators, seed_replacement, ratio, verbose=0):
    # Initializing list of random training sets
    X_sets = []
    y_sets = []

    ratio_size = int(len(X) * ratio)

    seed = None

    for i in range(0, n_estimators):  # for each new set
        instance_X = np.empty(np.shape(X), dtype=type(X[0]))
        instance_y = np.empty(np.shape(y), dtype=type(y[0]))

        # creating seed for specific training set
        seed = _pick_seed(K, seed, verbose)

        instance_X[0] = X[seed]
        instance_y[0] = y[seed]

        pool = np.arange(np.shape(X)[0])  # The "bingo sheet"
        pool_prob = np.zeros(np.shape(X)[0])  # Probability to be picked for each index in the sheet
        pool_tot = np.sum(K[seed])  # Sum of similarities with every point to the seed

        for j in range(len(X)):

            if K[seed][j] == 0 and not seed_replacement: # might break pool_tot?
                pool_prob[j] = 0
                if verbose > 1:
                    print("Un-chosen seed")
            else:
                pool_prob[j] = K[seed][j] / pool_tot

        # Extraction
        pick = np.random.choice(pool, size=ratio_size - 1, replace=True, p=pool_prob)

        if verbose > 2:
            print("Populating set", i, "...")

        for j in range(len(X)-1):
            try:
                instance_X[j + 1] = X[pick[j]]
                instance_y[j + 1] = y[pick[j]]
            except:
                index = randint(0, len(X)-1)
                instance_X[j + 1] = X[index]
                instance_y[j + 1] = y[index]

        X_sets.append(instance_X)
        y_sets.append(instance_y)

    return X_sets, y_sets


def _pick_seed(K, seed, verbose):
    # creating seed for specific training set, choosing by inverse similarity to previous set
    if seed is None:
        x = randint(0, len(K) - 1)

        if verbose > 1:
            print("Chose first seed", x)

        return x
    else:

        sim_n = 1 - K[seed]

        candidate = np.random.choice(np.arange(len(K)), size=1, replace=True,
                                     p=sim_n / np.sum(sim_n))[0]

        if verbose > 1:
            print("Chose new seed", candidate, "based on inverse similarity w/ previous one:", K[candidate][seed])

        return candidate


# Fills invalid training sets with elements of the missing classes, starting from the seeds.
def _autofill(X_sets, y_sets, X, y, n_classes, verbose):

    if verbose > 0:
        print(datetime.datetime.now().time(), "Filling up training sets with missing classes...")

    # Creating list of seeds to fill the sets
    X_seeds = [i[0] for i in X_sets]
    y_seeds = [i[0] for i in y_sets]

    classes = []
    for i in range(n_classes):
        classes.append(i)

    # Check if the set of seeds contain all classes. If not, search for missing classes
    # in the training set. Tedious, but mitigated by the fact that most of the
    # time this is needed on small training sets, which are fast to iterate.
    # Better than searching all classes anyway.
    if len(np.unique(y_seeds)) != n_classes:

        missing_classes = np.setxor1d(np.unique(y_seeds), classes)
        for i in missing_classes:
            for j in range(len(y)):
                if y[j] == i:
                    y_seeds.append(y[j])
                    X_seeds.append(X[j])

    for i in range(len(y_sets)):  # for each training set

        # calculate missing classes
        set_classes = np.unique(y_sets[i])
        missing_classes = np.setxor1d(set_classes, classes)

        # unique indexes to choose which elements to replace in the set
        indexes = random.sample(range(0, len(X_sets[0])), len(missing_classes))

        if len(set_classes) < n_classes:
            if verbose > 1:
                print("Filling set no.", i)
            for j in range(len(missing_classes)):
                for k in range(len(y_seeds)):
                    if y_seeds[k] == missing_classes[j]:
                        X_sets[i][indexes[int(j)]] = X_seeds[k]
                        y_sets[i][indexes[int(j)]] = y_seeds[k]

                        if verbose > 2:
                            print("Inserted value of class", y_seeds[k])
                        break

    return X_sets, y_sets


""" Custom exceptions """


class InvalidBootstrapError(Exception):
    """Raised when bootstrapped training set do not contain all classes."""
    pass


"""

    Parameters
    ----------
    n_estimators : int, default = 50
        The number of models to train.

    base_estimator : estimator, default = DecisionTreeClassifier()
        The estimator fitted on  each bootstrapped set.     
        
    K : ndarray, default = None
        Pre-computed Kernel matrix for fitting, useful for multiple folds.  
        
    ratio : float, default = 1
        Percentage of elements picked based on similarity during bootstrapping.

    metric : {'additive_chi2', 'chi2', 'linear', 'poly', 'polynomial', 'rbf', 'laplacian', 'sigmoid', 'cosine', 'custom'}, string, default="rbf"
        The metric used for pairwise_kernels().
        
    seed_replacement : boolean, default = True
        TODO
        Determines if seed can be picked again in the bootstrap(?)
        
    autofill : boolean, default = False
        Fixes bootstrapped training sets that do not contain all classes.
        Avoids raising the InvalidBootstrapError exception if set to True.

    verbose : int, default = 0
        Controls verbosity during fitting and predicting, 0 being none and 3 being the most detailed. 

"""


class TreeStrainsClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, n_jobs=1, n_estimators=50,
                 base_estimator=DecisionTreeClassifier(), ratio=1,
                 metric="rbf", seed_replacement=True, autofill=False,
                 verbose=0):

        self.n_jobs = n_jobs
        self.n_estimators = n_estimators
        self.base_estimator = base_estimator
        self.ratio = ratio
        self.metric = metric
        self.seed_replacement = seed_replacement
        self.autofill = autofill
        self.verbose = verbose
        self.X_sets = None
        self.y_sets = None
        self.X_ = None
        self.y_ = None
        self.estimators_ = None
        self.n_classes_ = None

        _validate_parameters(self.metric)

    def fit(self, X, y):

        self.n_classes_ = np.max(y) + 1

        if self.verbose > 0:
            print(datetime.datetime.now().time(), "Bootstrapping...")

        K = SimilarityFunction(X, metric=self.metric, n_jobs=self.n_jobs)

        if self.verbose > 1:
            print("Calculated similarity matrix of shape", np.shape(K))

        self.X_sets, self.y_sets = _similarity_bootstrap(X, y, K,
                                                         n_estimators=self.n_estimators,
                                                         seed_replacement=self.seed_replacement,
                                                         ratio=self.ratio,
                                                         verbose=self.verbose)

        if self.autofill:
            self.X_sets, self.y_sets = _autofill(self.X_sets, self.y_sets, X, y, self.n_classes_, self.verbose)

        if not _validate_bootstrap(self.y_sets, self.n_classes_, self.verbose):
            raise InvalidBootstrapError(
                'Some or all training sets do not contain at least one entry '
                'for each of the %s classes. Consider the following solutions:\n'
                '- re-run the fit() method(may take some attempts);\n'
                '- add more random elements by decreasing ratio;\n'
                '- use a more relevant similarity metric;\n'
                '- set autofill to True if not so already;\n'
                '- use a more balanced dataset.\n'
                'if none of these options work, consider using another classifier altogether.'
                % self.n_classes_
            )

        if self.verbose > 0:
            print(datetime.datetime.now().time(), "Generating models...")

        self.estimators_ = []
        for i in range(0, self.n_estimators):

            self.estimators_.append(self.base_estimator.fit(self.X_sets[i], self.y_sets[i]))

        self.X_ = X
        self.y_ = y

        if self.verbose > 0:
            print(datetime.datetime.now().time(), "Done!")
        return self

    def predict(self, X):

        predicted_probability = self.predict_proba(X)
        return np.argmax(predicted_probability, axis=1)

    def predict_proba(self, X, sim_matrix=False):

        if self.verbose > 0:
            print(datetime.datetime.now().time(), "Predicting...")

        out = np.zeros(shape=(len(X), self.n_classes_))
        K = []
        predictions = []
        sim_value = np.zeros(self.n_estimators)

        if self.verbose > 0:
            print(datetime.datetime.now().time(), "Computing predictions for each model...")

        for i in range(0, self.n_estimators):  # for each estimator trained

            if sim_matrix:
                indexes = random.sample(range(0, len(self.X_sets[0])), int(np.sqrt(len(self.X_sets[0]))))
                pool = []
                for j in range(len(indexes)):
                    pool.append(self.X_sets[i][indexes[j]])

                K.append(SimilarityFunction(X, Y=pool, metric=self.metric,
                                            n_jobs=self.n_jobs))

            else:
                K.append(SimilarityFunction(X, Y=self.X_sets[i][0].reshape(1, -1), metric=self.metric,
                                            n_jobs=self.n_jobs))

            predictions.append(self.estimators_[i].predict_proba(X))

        votes = np.zeros((len(X), len(predictions), self.n_classes_))

        if self.verbose > 0:
            print(datetime.datetime.now().time(), "Computing similarities between models and samples...")

        for i in range(0, len(X)):  # for each sample to predict

            if self.verbose > 2:
                print("\n--- EVALUATING SAMPLE", i, " ---")

            for j in range(0, self.n_estimators):  # for each model
                if sim_matrix:
                    sim_value[j] = np.mean(K[j][i])
                else:
                    sim_value[j] = K[j][i]

            if np.sum(sim_value) != 0:
                for j in range(0, self.n_estimators):
                    # weight votes
                    votes[i][j] = predictions[j][i] * ((100 * sim_value[j] / np.sum(sim_value)) / 100)

                    if self.verbose > 2:
                        print("model", j, "votes", np.argmax(votes[i][j]), "having ~", np.round(sim_value[j], 4),
                              "similarity (voting power:",
                              np.round(((100 * sim_value[j] / np.sum(sim_value)) / 100), 4),
                              ")")

                out[i] = (np.sum(votes[i], axis=0))
            else:
                warnings.warn("No similarity between input and training set. Using hard voting instead")
                for j in range(0, self.n_estimators):
                    votes[i][j] = predictions[j][i]

                    if self.verbose > 2:
                        print("model", j, "votes", np.argmax(votes[i][j]), "having ~", np.round(sim_value[j], 4),
                              "similarity")

                out[i] = (np.sum(votes[i], axis=0)) / self.n_estimators

                # index = np.argmax((np.sum(votes[i], axis=0)))
                #
                # out[i][index] = votes[i][j]

            if self.verbose > 1:
                print("SAMPLE", i, "PREDICTION:", out[i], )

        if self.verbose > 0:
            print(datetime.datetime.now().time(), "Done!")

        return out
