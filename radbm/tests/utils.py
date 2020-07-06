import unittest
from radbm.utils import unique_list

class Test_unique_list(unittest.TestCase):
    def test_unique_list(self):
        it = [3,4,4,2,1,1,0,3,2]
        expected_list = [3,4,2,1,0]
        result = unique_list(it)
        self.assertEqual(result, expected_list)