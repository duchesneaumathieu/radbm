import unittest, torch
import numpy as np
from radbm.search.elba import HashNet
from radbm.search.mbsds import HashingMultiBernoulliSDS

class TestHashNet(unittest.TestCase):
    def test_step(self):
        torch.manual_seed(0)
        model = HashNet(
            torch.nn.Linear(32,64),
            torch.nn.Linear(32,64),
            HashingMultiBernoulliSDS(1,1),
            1/32,
        )
        x = torch.eye(32, dtype=torch.float)
        r = torch.eye(32, dtype=torch.bool)
        l0 = model.step(x, x, r)
        for i in range(500):
            model.step(x, x, r)
        l1 = model.step(x, x, r)
        self.assertTrue(l1 < l0)