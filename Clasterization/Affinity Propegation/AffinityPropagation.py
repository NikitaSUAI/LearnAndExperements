# https://towardsdatascience.com/unsupervised-machine-learning-affinity-propagation-algorithm-explained-d1fef85f22c8

import numpy as np
from typing import Tuple


def SimilarityMatrix(data: np.array) -> np.array:
    # calculate similarity matrix
    # Note: mb later convert to functional style
    s = np.zeros((len(data), len(data)))
    for i in range(len(data)):
        for j in range(len(data)):
            s[i, j] = -((data[i] - data[j]) ** 2).sum()
    for i in range(len(data)):
        s[i, i] = s.min()
    return s


def ResponsibilityMatrix(s: np.array, a: np.array) -> np.array:
    # calculate responsibility matrix
    r = np.zeros(s.shape)
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            prep = ~np.isin(np.arange(s.shape[1]), [j, ])
            tmp = a[i, prep] + s[i, prep]
            r[i, j] = s[i, j] - np.max(tmp)
    return r


def AvailabilityMatrix(shape: Tuple, r: np.array = None) -> np.array:
    a = np.zeros(shape)
    if r is None:
        return a
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            prep = ~np.isin(np.arange(r.shape[0]), [i, j])
            tmp = np.maximum(np.zeros(r[prep, j].shape), r[prep, j]).sum()
            a[i, j] = np.min((0, r[j, j] + tmp)) if i != j else tmp
    return a


def Affinity_Propegation(data: np.array, iteration: int = 1):
    # define similarity matrix according to euclidean distance,
    # Responsibility matrix, Availability matrix and Criterion Matrix
    s = SimilarityMatrix(data)
    a = AvailabilityMatrix(s.shape)
    c = None
    for _ in range(iteration):
        r = ResponsibilityMatrix(s, a)
        a = AvailabilityMatrix(r.shape, r)
        c = r + a
    c = [np.argmax(i) for i in c]
    return c


if __name__ == "__main__":
    dataset = np.array([[3, 4, 3, 2, 1],
                        [4, 3, 5, 1, 1],
                        [3, 5, 3, 3, 3],
                        [2, 1, 3, 3, 2],
                        [1, 1, 3, 2, 3]])
    print(Affinity_Propegation(dataset, 1))
