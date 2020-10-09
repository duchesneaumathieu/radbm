import unittest, torch
import numpy as np
from radbm.search.elba import EfficientLearnableBinaryAccess
from radbm.search.mbsds import HashingMultiBernoulliSDS

class TestEfficientLearnableBinaryAccess(unittest.TestCase):
    def test_elba(self):
        f = torch.nn.Linear(784,128)
        elba = EfficientLearnableBinaryAccess(
            f, f, HashingMultiBernoulliSDS(1,1))
        data = np.random.RandomState(0xcafe).normal(0,1,(32, 784))
        data = torch.tensor(data, dtype=torch.float32)
        elba.batch_insert(data, range(32))
        #using the same function for fq and fd
        index = elba.batch_search(data)
        self.assertEqual(index, [{i} for i in range(32)])
        index = [next(g) for g in elba.batch_itersearch(data)]
        self.assertEqual(index, [{i} for i in range(32)])
        
    def test_get_and_set_state(self):
        f = torch.nn.Linear(784,128)
        elba = EfficientLearnableBinaryAccess(
            f, f, HashingMultiBernoulliSDS(1,1))
        elba.set_state(elba.get_state())