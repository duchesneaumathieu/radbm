import unittest
import numpy as np
from radbm.utils import unique_list, Ramp

class Test_unique_list(unittest.TestCase):
    def test_unique_list(self):
        it = [3,4,4,2,1,1,0,3,2]
        expected_list = [3,4,2,1,0]
        result = unique_list(it)
        self.assertEqual(result, expected_list)
        
class TestRamp(unittest.TestCase):
    def test_ramp(self):
        #from x=3..5 we ramp from y=-1..7
        ramp = Ramp(3, 5, -1, 7)
        xs = np.array([2, 3, 4, 5, 6])
        ys = np.array([-1, -1, 3, 7, 7])
        self.assertTrue(np.allclose(ramp(xs), ys))