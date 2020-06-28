import unittest
import numpy as np
from radbm.retrieval.base import Retrieval

class Dummy(Retrieval):
    pass

class SingleDummy(Retrieval):
    def __init__(self):
        self.index = dict()
    
    def insert(self, document, index):
        self.index[tuple(document)] = index
    
    def search(self, query):
        return self.index[tuple(query)]
    
class BatchDummy(Retrieval):
    def batch_insert(self, documents, indexes):
        self.indexes = [documents.shape, indexes]
    
    def batch_search(self, queries):
        return [self.indexes]
    
class TestRetrieval(unittest.TestCase):
    def test_NotImplementedError(self):
        dummy = Dummy()
        dummy_data = np.zeros((32,100))
        with self.assertRaises(NotImplementedError):
            dummy.insert(dummy_data[0], 0)
        with self.assertRaises(NotImplementedError):
            dummy.batch_insert(dummy_data, range(32))
        with self.assertRaises(NotImplementedError):
            dummy.search(dummy_data[0])
        with self.assertRaises(NotImplementedError):
            dummy.batch_search(dummy_data)
        with self.assertRaises(NotImplementedError):
            dummy.itersearch(dummy_data[0])
        with self.assertRaises(NotImplementedError):
            dummy.batch_itersearch(dummy_data[0])
            
    def test_default_insert(self):
        dummy = BatchDummy()
        dummy_data = np.zeros((100,))
        dummy.insert(dummy_data, 0)
        out = dummy.search(dummy_data)
        self.assertEqual(out, [(1,100), [0]])
        
    def test_default_batch_insert(self):
        dummy = SingleDummy()
        dummy_data = np.arange(32*100).reshape(32,100)
        dummy.batch_insert(dummy_data, range(32))
        out = dummy.batch_search(dummy_data)
        self.assertEqual(out, list(range(32)))