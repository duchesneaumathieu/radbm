import unittest
import numpy as np
from itertools import islice
from radbm.utils.generators import likeliest_multi_bernoulli_outcomes

class TestLikeliestMultiBernoulliOutcomes(unittest.TestCase):
    def test_likeliest_multi_bernoulli_outcomes(self):
        rng = np.random.RandomState(0xcafe)
        probs = rng.uniform(0, 1, 16)
        lp0 = np.log(1-probs)
        lp1 = np.log(probs)
        outcomes = list(islice(likeliest_multi_bernoulli_outcomes(lp0, lp1), 10))
        outcomes_nlp = [-lp1[outcome].sum() - lp0[~outcome].sum() for outcome in outcomes]
        
        #assert unique
        outcomes_tuple = [tuple(outcome) for outcome in outcomes]
        self.assertEqual(len(outcomes_tuple), len(set(outcomes_tuple)))
        
        #assert decreasing order
        self.assertTrue(np.all(0<=np.diff(outcomes_nlp)))
        self.assertEqual(len(outcomes), 10)
        
        #test yield_log_probs
        outcomes, log_probs = zip(*islice(likeliest_multi_bernoulli_outcomes(lp0, lp1, yield_log_probs=True), 10))
        self.assertTrue(np.allclose(outcomes_nlp, [-lp for lp in log_probs]))
        
        #test yield_stats
        self.assertEqual(4, len(next(likeliest_multi_bernoulli_outcomes(lp0, lp1, yield_stats=True))))
        
        #test ndim != 1 error
        with self.assertRaises(ValueError):
            next(likeliest_multi_bernoulli_outcomes(lp0[None], lp1))
        with self.assertRaises(ValueError):
            next(likeliest_multi_bernoulli_outcomes(lp0, lp1[None]))
        
        #test not summing to one error
        with self.assertRaises(ValueError):
            next(likeliest_multi_bernoulli_outcomes(lp0, lp0))