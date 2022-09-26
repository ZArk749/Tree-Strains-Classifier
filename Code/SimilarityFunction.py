from sklearn.metrics import pairwise_kernels


def SimilarityFunction(X, Y=None, metric='', n_jobs=1):
    if metric in ("rbf", "laplacian"):
        if Y is None:
            return pairwise_kernels(X, metric=metric, n_jobs=n_jobs)
        else:
            return pairwise_kernels(X, Y, metric=metric, n_jobs=n_jobs)
    else:
        # put here your custom similarity function
        raise ValueError("Custom similarity metric '%s' not defined" % metric)
