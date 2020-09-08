import unittest, torch
import numpy as np
from radbm.search.elbm import EfficientLearnableBinaryMemory
from radbm.search.mbsds import HashingMultiBernoulliSDS

#need test multi_bernoulli_*

class TestEfficientLearnableBinaryMemory(unittest.TestCase):
    def test_elbm(self):
        f = torch.nn.Linear(784,128)
        elbm = EfficientLearnableBinaryMemory(
            f, f, HashingMultiBernoulliSDS(1,1))
        data = np.random.RandomState(0xcafe).normal(0,1,(32, 784))
        data = torch.tensor(data, dtype=torch.float32)
        elbm.batch_insert(data, range(32))
        #using the same function for fq and fd
        index = elbm.batch_search(data)
        self.assertEqual(index, [{i} for i in range(32)])
        index = [next(g) for g in elbm.batch_itersearch(data)]
        self.assertEqual(index, [{i} for i in range(32)])
        
test = TestEfficientLearnableBinaryMemory()
test.test_elbm()