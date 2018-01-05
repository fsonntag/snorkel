import re
import sys

import numpy as np
import scipy.sparse as sparse


class ProgressBar(object):
    def __init__(self, N, length=40):
        # Protect against division by zero (N = 0 results in full bar being printed)
        self.N = max(1, N)
        self.nf = float(self.N)
        self.length = length
        # Precalculate the i values that should trigger a write operation
        self.ticks = set([round(i / 100.0 * N) for i in range(101)])
        self.ticks.add(N - 1)
        self.bar(0)

    def bar(self, i):
        """Assumes i ranges through [0, N-1]"""
        if i in self.ticks:
            b = int(np.ceil(((i + 1) / self.nf) * self.length))
            sys.stdout.write(
                "\r[{0}{1}] {2}%".format(
                    "=" * b, " " * (self.length - b), int(100 * ((i + 1) / self.nf))))
            sys.stdout.flush()

    def close(self):
        # Move the bar to 100% before closing
        self.bar(self.N - 1)
        sys.stdout.write("\n\n")
        sys.stdout.flush()


def get_ORM_instance(ORM_class, session, instance):
    """
    Given an ORM class and *either an instance of this class, or the name attribute of an instance
    of this class*, return the instance
    """
    if isinstance(instance, str):
        return session.query(ORM_class).filter(ORM_class.name == instance).one()
    else:
        return instance


def camel_to_under(name):
    """
    Converts camel-case string to lowercase string separated by underscores.

    Written by epost
    (http://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case).

    :param name: String to be converted
    :return: new String with camel-case converted to lowercase, underscored
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def sparse_abs(X):
    """Element-wise absolute value of sparse matrix- avoids casting to dense matrix!"""
    X_abs = X.copy()
    if not sparse.issparse(X):
        return abs(X_abs)
    if sparse.isspmatrix_csr(X) or sparse.isspmatrix_csc(X):
        X_abs.data = np.abs(X_abs.data)
    elif sparse.isspmatrix_lil(X):
        X_abs.data = np.array([np.abs(L) for L in X_abs.data])
    else:
        raise ValueError("Only supports CSR/CSC and LIL matrices")
    return X_abs


def matrix_coverage(L):
    """
    Given an N x M matrix where L_{i,j} is the label given by the jth LF to the ith candidate:
    Return the **fraction of candidates that each LF labels.**
    """
    return np.ravel(sparse_abs(L).sum(axis=0) / float(L.shape[0]))


def matrix_overlaps(L):
    """
    Given an N x M matrix where L_{i,j} is the label given by the jth LF to the ith candidate:
    Return the **fraction of candidates that each LF _overlaps with other LFs on_.**
    """
    L_abs = sparse_abs(L)
    return np.ravel(np.where(L_abs.sum(axis=1) > 1, 1, 0).T * L_abs / float(L.shape[0]))


def matrix_conflicts(L):
    """
    Given an N x M matrix where L_{i,j} is the label given by the jth LF to the ith candidate:
    Return the **fraction of candidates that each LF _conflicts with other LFs on_.**
    """
    L_abs = sparse_abs(L)
    return np.ravel(np.where(L_abs.sum(axis=1) != sparse_abs(L.sum(axis=1)), 1, 0).T * L_abs / float(L.shape[0]))


def matrix_tp(L, labels):
    false_value = int(L.max())
    return np.ravel(np.sum([[
        np.sum(np.ravel((L[:, j] == i).todense()) * (labels == i)) for j in range(L.shape[1])
    ] for i in range(1, false_value)], axis=0))


def matrix_fp(L, labels):
    false_value = int(L.max())
    all_values = set(range(false_value + 1))
    return np.ravel(np.sum([np.sum([[
        np.sum(np.ravel((L[:, j] == k).todense()) * (labels == i)) for j in range(L.shape[1])
    ] for i in all_values - {k}], axis=0) for k in range(1, false_value)], axis=0))


def matrix_tn(L, labels):
    false_value = int(L.max())
    return np.ravel([
        np.sum(np.ravel((L[:, j] == false_value).todense()) * (labels == 0)) for j in range(L.shape[1])
    ])


def matrix_fn(L, labels):
    false_value = int(L.max())
    return np.ravel(np.sum([[
        np.sum(np.ravel((L[:, j] == false_value).todense()) * (labels == i)) for j in range(L.shape[1])
    ] for i in range(1, false_value)], axis=0))


def get_as_dict(x):
    """Return an object as a dictionary of its attributes"""
    if isinstance(x, dict):
        return x
    else:
        try:
            return x._asdict()
        except AttributeError:
            return x.__dict__


def sort_X_on_Y(X, Y):
    return [x for (y, x) in sorted(zip(Y, X), key=lambda t: t[0])]


def corenlp_cleaner(words):
    d = {'-RRB-': ')', '-LRB-': '(', '-RCB-': '}', '-LCB-': '{',
         '-RSB-': ']', '-LSB-': '['}
    return map(lambda w: d[w] if w in d else w, words)


def tokens_to_ngrams(tokens, n_max=3, delim=' '):
    N = len(tokens)
    for root in range(N):
        for n in range(min(n_max, N - root)):
            yield delim.join(tokens[root:root + n + 1])


def overlapping_score(candidate1, candidate2):
    span1 = candidate1[0]
    span2 = candidate2[0]
    if span1.sentence_id != span2.sentence_id:
        return 0.
    start1, end1 = span1.char_start, span1.char_end
    start2, end2 = span2.char_start, span2.char_end
    if start1 == start2 and end1 == end2:
        return 1.
    if not (start1 <= end2
            and end1 >= start2):
        return 0.
    if end1 == end2:
        common_chars = end1 - start2
        non_common_chars = start1 - start2
    elif end2 > end1:
        common_chars = (end1 + 1) - max(start1, start2)
        non_common_chars = end2 - end1
    else:
        common_chars = (end2 + 1) - max(start1, start2)
        non_common_chars = start1 - start2
    if non_common_chars < 0:
        non_common_chars = 0
    length1 = (end1 + 1) - start1
    length2 = (end2 + 1) - start2
    return abs(common_chars / length1 * (1 - non_common_chars / length2))
