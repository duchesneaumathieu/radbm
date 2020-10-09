import unittest, torch
import numpy as np
from radbm.search.elba import Fbeta
from radbm.search.mbsds import HashingMultiBernoulliSDS

class TestFbeta(unittest.TestCase):
    def test_step(self):
        torch.manual_seed(0)
        model = Fbeta(
            torch.nn.Linear(32,16),
            torch.nn.Linear(32,16),
            HashingMultiBernoulliSDS(1,1),
            -np.log(32), # log(1/bs)
            match_dist=0,
        )
        x = torch.eye(32, dtype=torch.float)
        r = torch.eye(32, dtype=torch.bool)
        for i in range(500):
            model.step(x, x, r)
        model.batch_insert(x, range(32))
        out = model.batch_search(x)
        self.assertEqual(out, [{i} for i in range(32)])