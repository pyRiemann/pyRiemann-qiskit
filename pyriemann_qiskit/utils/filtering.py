from sklearn.base import TransformerMixin
import itertools
import numpy as np
from sklearn.svm import SVC

class NoFilter(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X

class NaivePair(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[:, ::2]

class NaiveImpair(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[:, 1::2]

class Closest(TransformerMixin):
    def __init__(self, ndim=None):
        self.ndim = ndim
        self.closest_subset_to_vector = None

    def fit(self, X, y=None):
        n_ts = X.shape[1]
        if self.ndim is None:
            self.ndim = n_ts // 2
        indices = range(n_ts)
        ret = {}
        print(indices, self.ndim)
        for v in X:
            for subset in itertools.combinations(indices, self.ndim):
                not_indices = [i for i in indices if i not in subset]
                sub_v = v.copy()
                sub_v[not_indices] = 0
                dist = np.linalg.norm(v - sub_v)
                key = ''.join(str(val) for val in subset)
                if key in ret:
                    ret[key] = ret[key] + dist
                else:
                    ret[key] = dist
        self.closest_subset_to_vector = min(ret, key=ret.get)
        print("Fit finished")
        return self

    def transform(self, X, y=None):
        subset = [int(i) for i in self.closest_subset_to_vector]
        ret = X[:, subset]
        return ret


class Preclassif(TransformerMixin):
    def __init__(self, ndim=None):
        self.ndim = ndim
        self.closest_subset_to_vector = None

    def fit(self, X, y=None):
        n_ts = X.shape[1]
        if self.ndim is None:
            self.ndim = n_ts // 2
        indices = range(n_ts)
        ret = {}
        for subset in itertools.combinations(indices, self.ndim):
            sub_vectors = None
            for v in X:
                if sub_vectors is None:
                    sub_vectors = np.atleast_2d([v[list(subset)]])
                else:
                    sub_vectors = np.append(sub_vectors, [v[list(subset)]], axis=0)
            svc = SVC()

            svc.fit(sub_vectors, y)
            key = ''.join(str(val) for val in subset)
            ret[key] = svc.score(sub_vectors, y)
        self.closest_subset_to_vector = max(ret, key=ret.get)
        print("Fit finished")
        return self

    def transform(self, X, y=None):
        subset = [int(i) for i in self.closest_subset_to_vector]
        ret = X[:, subset]
        return ret

