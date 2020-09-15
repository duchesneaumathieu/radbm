import unittest, torch
import numpy as np
from radbm.search.elba import (
    EfficientLearnableBinaryAccess,
    multi_bernoulli_equality,
    multi_bernoulli_subset,
    multi_bernoulli_activated_equality,
    multi_bernoulli_activated_subset,
)
from radbm.search.mbsds import HashingMultiBernoulliSDS

class TestMultiBernoulliOperators(unittest.TestCase):
    def test_multi_bernoulli_equality(self):
        x = torch.tensor([-.5, .1, .9])
        y = torch.tensor([.5, .3, -.8])
        sig_x = torch.sigmoid(x)
        sig_y = torch.sigmoid(y)
        expected = sig_x*sig_y + (1-sig_x)*(1-sig_y)
        z = multi_bernoulli_equality(x, y)
        self.assertTrue(torch.allclose(z.exp(), expected))
    
    def test_multi_bernoulli_subset(self):
        x = torch.tensor([-.5, .1, .9])
        y = torch.tensor([.5, .3, -.8])
        sig_x = torch.sigmoid(x)
        sig_y = torch.sigmoid(y)
        expected = sig_x*sig_y + (1-sig_x)*(1-sig_y) + (1-sig_x)*sig_y
        z = multi_bernoulli_subset(x, y)
        self.assertTrue(torch.allclose(z.exp(), expected))
    
    def test_multi_bernoulli_activated_equality(self):
        x = torch.tensor([-.5, .1, .9])
        y = torch.tensor([.5, .3, -.8])
        a = torch.tensor([.5, .3, -.8])
        sig_x = torch.sigmoid(x)
        sig_y = torch.sigmoid(y)
        sig_a = torch.sigmoid(a)
        equal = sig_x*sig_y + (1-sig_x)*(1-sig_y)
        expected = sig_a + (1-sig_a)*equal
        z = multi_bernoulli_activated_equality(x, y, a)
        self.assertTrue(torch.allclose(z.exp(), expected))
    
    def test_multi_bernoulli_activated_subset(self):
        x = torch.tensor([-.5, .1, .9])
        y = torch.tensor([.5, .3, -.8])
        a = torch.tensor([.5, .3, -.8])
        sig_x = torch.sigmoid(x)
        sig_y = torch.sigmoid(y)
        sig_a = torch.sigmoid(a)
        subset = sig_x*sig_y + (1-sig_x)*(1-sig_y) + (1-sig_x)*sig_y
        expected = sig_a + (1-sig_a)*subset
        z = multi_bernoulli_activated_subset(x, y, a)
        self.assertTrue(torch.allclose(z.exp(), expected))

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