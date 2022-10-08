from sklearn.metrics import pairwise_kernels


def SimilarityFunction(X, Y=None, metric='', n_jobs=1):
    if metric == "custom":
        # put here your custom similarity functions
        if Y is None:
            K = pairwise_kernels(X, metric="chi2", n_jobs=n_jobs)
        else:
            K = pairwise_kernels(X, Y, metric="chi2", n_jobs=n_jobs)
        return K
        raise ValueError("Custom similarity metric '%s' not defined" % metric)
    else:
        if Y is None:
            return pairwise_kernels(X, metric=metric, n_jobs=n_jobs)
        else:
            return pairwise_kernels(X, Y, metric=metric, n_jobs=n_jobs)