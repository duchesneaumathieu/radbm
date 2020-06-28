import unittest
import numpy as np

from radbm.metrics.oracle import *
rng = np.random.RandomState(0xcafe)

#fake model
class Model(object):
    def __init__(self, N, relevant_sets, precision):
        self.N = N
        self.relevant_sets = relevant_sets
        self.precision = precision #precision
        
    def candidates(self, q):
        rs = iter(self.relevant_sets[q])
        for i in range(self.N+1):
            if rng.uniform(0,1) < self.precision:
                try: yield [next(rs)]
                except StopIteration: break
            else: yield [self.N+1] #not in relevant_set

class TestOracleMetric(unittest.TestCase):
    def query_iterator(self, M, relevant_sets):
        for i in range(M):
            yield i, relevant_sets[i]
            
    def test_basic_oracle_metric(self):
        N = 10000
        M = 100
        k = 5
        relevant_sets = [set(rng.randint(0,N,k)) for _ in range(M)]
        model_precision = .7
        query_iterator = self.query_iterator(M, relevant_sets)
        model = Model(N, relevant_sets, model_precision)
        ToCRs = time_over_oracle_calls_reduction_metric(
            model, query_iterator, N, recalls=[.1,.25,.5,1])
        
    def test_bad_candidates_oracle_metric(self):
        N = 10000
        M = 100
        k = 5
        relevant_sets = [set(rng.randint(0,N,k)) for _ in range(M)]
        model_precision = .00001 #the model will fail to achieve 1 recall
        query_iterator = self.query_iterator(M, relevant_sets)
        model = Model(N, relevant_sets, model_precision)
        with self.assertRaises(RuntimeError):
            ToCRs = time_over_oracle_calls_reduction_metric(
                model, query_iterator, N, recalls=[.1,.25,.5,1])
