import unittest, torch
from radbm.utils.torch import HuberLoss

class TestHuberLoss(unittest.TestCase):
    def test_huber_loss(self):
        #empty test, just make sure it runs.
        x = torch.linspace(-10, 10, 200)
        y = HuberLoss(2, 5)(x)