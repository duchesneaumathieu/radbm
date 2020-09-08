import unittest
import numpy as np
from scipy.stats import hypergeom

from radbm.utils.stats import *

np.seterr(all='raise')
def logsigmoid(x): return -np.log(1 + np.exp(-x))

class Test_least_k_subset_sum_generator(unittest.TestCase):
    def test_least_k_subset_sum_generator(self):
        rng = np.random.RandomState(0xcafe)
        values = rng.uniform(0, 100, (1000,))
        gen = least_k_subset_sum_generator(values, k=10**4)
        sums = [sum(values[list(subset)]) for subset in gen]
        self.assertTrue(np.all(0<=np.diff(sums))) #assert increasing order
        self.assertTrue(len(sums)==10**4)
        
        #assert generating all
        values = rng.uniform(0, 100, (10,))
        gen = least_k_subset_sum_generator(values)
        self.assertTrue(2**10, len(list(gen)))
    
def logsigmoid(x): return -np.log(1 + np.exp(-x))
class Test_greatest_k_multi_bernoulli_outcomes_generator(unittest.TestCase):
    def compute_outcomes_prob(self, generator, log_probs0, log_probs1):
        probs = list()
        for bits in generator:
            bits = np.array(bits)
            log_probs = np.where(bits, log_probs1, log_probs0)
            probs.append(np.exp(log_probs.sum()))
        return probs
            
    def test_multi_bernoulli_top_k_generato(self):
        rng = np.random.RandomState(0xcafe)
        logits = rng.uniform(-10, 10, (8,))
        log_probs0 = logsigmoid(logits)
        log_probs1 = logsigmoid(-logits)

        gen1 = greatest_k_multi_bernoulli_outcomes_generator(log_probs0, log_probs1, k=200)
        gen2 = greatest_k_multi_bernoulli_outcomes_generator(log_probs0, k=200)
        probs1 = self.compute_outcomes_prob(gen1, log_probs0, log_probs1)
        probs2 = self.compute_outcomes_prob(gen2, log_probs0, log_probs1)
        self.assertEqual(probs1, probs2)
        self.assertTrue(np.all(np.diff(probs1)<=0)) #assert descending order
        self.assertEqual(200, len(probs1)) #assert we generate k outcomes
        
        #assert that we generate all outcomes when k is None
        self.assertEqual(2**8, len(list(greatest_k_multi_bernoulli_outcomes_generator(log_probs0))))
        
        logits = rng.uniform(-100, 100, (8,))
        log_probs0 = logsigmoid(logits)
        log_probs1 = logsigmoid(-logits)
        gen1 = greatest_k_multi_bernoulli_outcomes_generator(log_probs0, log_probs1, k=200)
        gen2 = greatest_k_multi_bernoulli_outcomes_generator(log_probs0, k=200)
        next(gen1) #runs
        with self.assertRaises(FloatingPointError):
            #numerically unstable when log_probs1 is not given
            next(gen2)

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