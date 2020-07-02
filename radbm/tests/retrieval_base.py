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
    
    def itersearch(self, query):
        for v in self.index.values():
            yield v
    
class BatchDummy(Retrieval):
    def batch_insert(self, documents, indexes):
        self.indexes = [documents.shape, indexes]
    
    def batch_search(self, queries):
        return [self.indexes]
    
    def batch_itersearch(self, query):
        all_indexes = np.arange(32)
        for indexes in all_indexes.reshape(-1, 8): #batch of size 6
            yield indexes
    
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
            next(dummy.itersearch(dummy_data[0]))
        with self.assertRaises(NotImplementedError):
            next(dummy.batch_itersearch(dummy_data[0]))
            
    def test_default_insert(self):
        dummy = BatchDummy()
        dummy_data = np.zeros((100,))
        dummy.insert(dummy_data, 0)
        out = dummy.search(dummy_data)
        self.assertEqual(out, [(1,100), [0]])
        out = list(dummy.itersearch(dummy_data))
        self.assertEqual(out, list(range(32)))
        
    def test_default_batch_insert(self):
        dummy = SingleDummy()
        dummy_data = np.arange(32*100).reshape(32,100)
        dummy.batch_insert(dummy_data, range(32))
        out = dummy.batch_search(dummy_data)
        self.assertEqual(out, list(range(32)))
        out = list(dummy.batch_itersearch(dummy_data))
        self.assertEqual(out, [[i] for i in range(32)])