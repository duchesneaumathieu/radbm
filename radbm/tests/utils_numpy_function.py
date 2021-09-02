import unittest
import numpy as np
from scipy.special import comb
from radbm.utils.numpy.function import log_comb, softplus, softplusinv, dihedral4, numpy_log_sigmoid

class TestNumpyFunction(unittest.TestCase):
    def test_log_comb(self):
        n_list = [0, 10, 10, 10, 30, 100, 100]
        k_list = [0,  4,  0, 10, 19,  10,  85]
        places = [7,  7,  7,  7,  7,   1,  -4] #numerical stability
        for n, k, p in zip(n_list, k_list, places):
            self.assertAlmostEqual(np.exp(log_comb(n, k)), comb(n, k), places=p)
            
    def test_softplus(self):
        rng = np.random.RandomState(0xcafe)
        x = rng.normal(0, 3, (2, 3, 4, 5))
        expected_sp = np.log(np.exp(x) + 1)
        sp = softplus(x)
        x_ = softplusinv(sp)
        np.testing.assert_allclose(expected_sp, sp)
        np.testing.assert_allclose(x, x_)
        
    def test_dihedral4(self):
        x = np.arange(4).reshape(2, 2) #[[0, 1], [2, 3]]
        self.assertEqual(dihedral4(x, 'r0').tolist(), [[0, 1], [2, 3]])
        self.assertEqual(dihedral4(x, 'r1').tolist(), [[1, 3], [0, 2]])
        self.assertEqual(dihedral4(x, 'r2').tolist(), [[3, 2], [1, 0]])
        self.assertEqual(dihedral4(x, 'r3').tolist(), [[2, 0], [3, 1]])
        self.assertEqual(dihedral4(x, 'sr0').tolist(), [[1, 0], [3, 2]])
        self.assertEqual(dihedral4(x, 'sr1').tolist(), [[3, 1], [2, 0]])
        self.assertEqual(dihedral4(x, 'sr2').tolist(), [[2, 3], [0, 1]])
        self.assertEqual(dihedral4(x, 'sr3').tolist(), [[0, 2], [1, 3]])
        
        with self.assertRaises(ValueError):
            dihedral4(x, 'bad input')
            
    def test_numpy_log_sigmoid(x):
        x = np.linspace(-10, 10, 1000)
        log_sig = numpy_log_sigmoid(x)
        sig = 1 / (1 + np.exp(-x))
        np.allclose(log_sig, np.log(sig))