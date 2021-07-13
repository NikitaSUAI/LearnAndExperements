from copy import copy

import AffinityPropagation
import unittest
import numpy as np


def template(func, args, result):
    return np.array(
        [np.equal(func(*p), r).all() for p, r in zip(args, result)])


class AffinityPropagationTestCase(unittest.TestCase):
    def __init__(self, *args):
        super().__init__(*args)
        self.data = [np.array([[3, 4, 3, 2, 1],
                               [4, 3, 5, 1, 1],
                               [3, 5, 3, 3, 3],
                               [2, 1, 3, 3, 2],
                               [1, 1, 3, 2, 3]]), ]
        self.similarity_result = [np.array([[-22, -7, -6, -12, -17],
                                            [-7, -22, -17, -17, -22],
                                            [-6, -17, -22, -18, -21],
                                            [-12, -17, -18, -22, -3],
                                            [-17, -22, -21, -3, -22]]), ]
        self.responsibility_result = [np.array([[-16, -1, 1, -6, -11],
                                                [10, -15, -10, -10, -15],
                                                [11, -11, -16, -12, -15],
                                                [-9, -14, -15, -19, 9],
                                                [-14, -19, -18, 14, -19]]), ]
        self.availability_result = [np.array([[21, -15, -16,  -5, -10],
                                             [-5,   0, -15,  -5, -10.],
                                             [-6, -15,   1,  -5, -10.],
                                             [0,  -15, -15,  14, -19.],
                                             [0,  -15, -15, -19,   9.]]), ]
        self.affinity_propagation_result = [np.array([0, 0, 0, 3, 3]), ]

    def test_similarity_matrix(self):
        args = [(i, ) for i in self.data]
        tests = template(AffinityPropagation.SimilarityMatrix, args,
                              self.similarity_result)
        self.assertTrue(tests.all())

    def test_responsibility_matrix(self):
        similarity_arg = copy(self.similarity_result)
        availability_arg = [np.zeros(similarity_arg[0].shape), ]
        args = [(j, i) for i, j in zip(availability_arg, similarity_arg)]
        tests = template(AffinityPropagation.ResponsibilityMatrix, args,
                         self.responsibility_result)
        self.assertTrue(tests.all())

    def test_availability_matrix(self):
        args = [((3, 3), None), ] + [(i.shape, i) for i in
                                     self.responsibility_result]
        res = [np.zeros((3, 3)), ] + self.availability_result
        tests = template(AffinityPropagation.AvailabilityMatrix, args, res)
        self.assertTrue(tests.all())

    def test_affinity_propagation(self):
        args = [(i, ) for i in self.data]
        tests = template(AffinityPropagation.Affinity_Propegation, args,
                         self.affinity_propagation_result)
        self.assertTrue(tests.all())


if __name__ == '__main__':
    unittest.main()
