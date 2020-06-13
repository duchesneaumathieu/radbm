import unittest
import numpy as np
from radbm.utils.torch import torch_cast_cpu

class TorchCast(unittest.TestCase):
    def test_torch_cast_cpu(self):
        a = np.zeros((4,5), dtype=np.float32)
        b = torch_cast_cpu(a)
        self.assertTrue((a==b.numpy()).all())
