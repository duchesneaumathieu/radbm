import unittest
from radbm.search.reduction.base import PointwiseReduction

class TestPointwiseReduction(unittest.TestCase):
    def test_notimplementederror(self):
        with self.assertRaises(NotImplementedError):
            PointwiseReduction('struct').queries_reduction(0)
        with self.assertRaises(NotImplementedError):
            PointwiseReduction('struct').documents_reduction(0)