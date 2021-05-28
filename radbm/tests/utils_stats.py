import unittest
import numpy as np
from scipy.stats import hypergeom

from radbm.utils.stats import (
    hypergeometric,
    superdupergeometric,
    superdupergeometric_expectations,
)

np.seterr(all='raise')

class TestHypergeometric(unittest.TestCase):
    def test_hypergeometric(self):
        N = 20
        K = 3
        p = hypergeometric(N, K)
        scipy_p = np.array([hypergeom(N, K, n).pmf(range(0,K+1)) for n in range(N+1)])
        err = np.abs(p-scipy_p).max()
        self.assertTrue(err < 1e-10)

    def test_superdupergeometric_and_expectations(self):
        N = 10000
        K = 125
        sp = superdupergeometric(N, K)
        a = (sp*np.arange(N+1)[:,None]).sum(axis=0)
        b = b = superdupergeometric_expectations(N, K)
        err = np.abs(a[1:] - b[1:]).max()
        self.assertTrue(err < 1e-8)