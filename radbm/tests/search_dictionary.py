import unittest
from radbm.search import DictionarySearch

class TestDictionarySearch(unittest.TestCase):
    def test_dictionary_search(self):
        dcodes = (123, 245, 12, 723, 120, 123, 12, 12)
        #           0,   1,  2,   3,   4,   5,  6,  7
        indexes = range(len(dcodes))
        cds = DictionarySearch().batch_insert(dcodes, indexes)
        cds = DictionarySearch().set_state(cds.get_state()) #testing set/get_state
        qcodes = (723, 120, 12, 123, 723, 42, 245)
        retrieved = cds.batch_search(qcodes)
        expected_retrieved = [
            {3}, #723
            {4}, #120
            {2, 6, 7}, #12
            {0, 5}, #123
            {3}, #723 (again)
            set(), #42
            {1}, #245
        ]
        self.assertEqual(retrieved, expected_retrieved)