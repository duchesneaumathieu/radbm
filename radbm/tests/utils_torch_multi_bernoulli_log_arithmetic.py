import unittest, torch
import numpy as np
from radbm.utils.torch import (
    multi_bernoulli_equality,
    multi_bernoulli_subset,
    multi_bernoulli_activated_equality,
    multi_bernoulli_activated_subset,
)

class TestMultiBernoulliLogArithmetic(unittest.TestCase):
    def test_multi_bernoulli_equality(self):
        x = torch.tensor([-.5, .1, .9])
        y = torch.tensor([.5, .3, -.8])
        sig_x = torch.sigmoid(x)
        sig_y = torch.sigmoid(y)
        expected = sig_x*sig_y + (1-sig_x)*(1-sig_y)
        z0, z1 = multi_bernoulli_equality(x, y)
        self.assertTrue(torch.allclose(z1.exp(), expected))
        self.assertTrue(torch.allclose(z0.exp()+z1.exp(), torch.tensor(1.)))
    
    def test_multi_bernoulli_subset(self):
        x = torch.tensor([-.5, .1, .9])
        y = torch.tensor([.5, .3, -.8])
        sig_x = torch.sigmoid(x)
        sig_y = torch.sigmoid(y)
        expected = sig_x*sig_y + (1-sig_x)*(1-sig_y) + (1-sig_x)*sig_y
        z0, z1 = multi_bernoulli_subset(x, y)
        self.assertTrue(torch.allclose(z1.exp(), expected))
        self.assertTrue(torch.allclose(z0.exp()+z1.exp(), torch.tensor(1.)))
    
    def test_multi_bernoulli_activated_equality(self):
        x = torch.tensor([-.5, .1, .9])
        y = torch.tensor([.5, .3, -.8])
        a = torch.tensor([.5, .3, -.8])
        sig_x = torch.sigmoid(x)
        sig_y = torch.sigmoid(y)
        sig_a = torch.sigmoid(a)
        equal = sig_x*sig_y + (1-sig_x)*(1-sig_y)
        expected = sig_a + (1-sig_a)*equal
        z0, z1 = multi_bernoulli_activated_equality(x, y, a)
        self.assertTrue(torch.allclose(z1.exp(), expected))
        self.assertTrue(torch.allclose(z0.exp()+z1.exp(), torch.tensor(1.)))
    
    def test_multi_bernoulli_activated_subset(self):
        x = torch.tensor([-.5, .1, .9])
        y = torch.tensor([.5, .3, -.8])
        a = torch.tensor([.5, .3, -.8])
        sig_x = torch.sigmoid(x)
        sig_y = torch.sigmoid(y)
        sig_a = torch.sigmoid(a)
        subset = sig_x*sig_y + (1-sig_x)*(1-sig_y) + (1-sig_x)*sig_y
        expected = sig_a + (1-sig_a)*subset
        z0, z1 = multi_bernoulli_activated_subset(x, y, a)
        self.assertTrue(torch.allclose(z1.exp(), expected))
        self.assertTrue(torch.allclose(z0.exp()+z1.exp(), torch.tensor(1.)))