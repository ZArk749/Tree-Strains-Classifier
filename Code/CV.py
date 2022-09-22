# Cross Validation
import time
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split


def CrossValidation(classifier, X, y):
    start_time = time.time()

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

    AccArrayCV = []
    BalAccArrayCV = []
    MicroF1ArrayCV = []
    MacroF1ArrayCV = []
    ExecutionTimeCV = []

    for train_index, test_index in skf.split(X, y):
        X_train = X[train_index]
        X_test = X[test_index]
        y_train = y[train_index]
        y_test = y[test_index]

        X_train_val, X_test_val, y_train_val, y_test_val = train_test_split(
            X_train, y_train, test_size=0.33, random_state=42)

        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        AccArrayCV.append(accuracy_score(y_test, y_pred))
        BalAccArrayCV.append(balanced_accuracy_score(y_test, y_pred))
        MicroF1ArrayCV.append(f1_score(y_test, y_pred, average='micro'))
        MacroF1ArrayCV.append(f1_score(y_test, y_pred, average='macro'))
        ExecutionTimeCV.append(time.time() - start_time)
        start_time = time.time()

    executionTimeCV = np.mean(ExecutionTimeCV)
    totalTimeCV = np.sum(ExecutionTimeCV)
    accCV = np.mean(AccArrayCV)
    balaccCV = np.mean(BalAccArrayCV)
    microf1CV = np.mean(MicroF1ArrayCV)
    macrof1CV = np.mean(MacroF1ArrayCV)

    return {
        'acc': accCV,
        'balacc': balaccCV,
        'microf1': microf1CV,
        'macrof1': macrof1CV,
        'time': totalTimeCV,
    }