import unittest
import numpy as np
from radbm.search.base import BaseSDS

class SingleDummy(BaseSDS):
    def __init__(self):
        self.index = dict()
    
    def insert(self, document, index):
        td = tuple(document)
        if td in self.index: self.index[td].add(index)
        else: self.index[td] = {index}
    
    def search(self, query):
        return self.index[tuple(query)]
    
    def itersearch(self, query):
        #yield everything
        yield from self.index.values()
    
class BatchDummy(BaseSDS):
    def __init__(self):
        self.index = dict()
    
    def batch_insert(self, documents, indexes):
        for d, i in zip(documents, indexes):
            td = tuple(d)
            if td in self.index: self.index[td].add(i)
            else: self.index[td] = {i}
    
    def batch_search(self, queries):
        return [self.index[tuple(query)] for query in queries]
    
    def batch_itersearch(self, queries):
        #yield everything for every queries
        return [self.index.values() for query in queries]
    
class TestBaseSDS(unittest.TestCase):
    def test_NotImplementedError(self):
        dummy = BaseSDS()
        with self.assertRaises(NotImplementedError):
            dummy.insert(123, 123)
        with self.assertRaises(NotImplementedError):
            dummy.batch_insert(123, 123)
        with self.assertRaises(NotImplementedError):
            dummy.search(123)
        with self.assertRaises(NotImplementedError):
            dummy.batch_search(123)
        with self.assertRaises(NotImplementedError):
            next(dummy.itersearch(123))
        with self.assertRaises(NotImplementedError):
            dummy.batch_itersearch(123)
            
    def test_default_single(self):
        dummy = BatchDummy()
        dummy_data = np.zeros((100,))
        dummy.insert(dummy_data, 0)
        out = dummy.search(dummy_data)
        self.assertEqual(out, {0})
        out = list(dummy.itersearch(dummy_data))
        self.assertEqual(out, [{0}])
        
    def test_default_batch(self):
        dummy = SingleDummy()
        dummy_data = np.arange(32*100).reshape(32,100)
        dummy.batch_insert(dummy_data, range(32))
        out = dummy.batch_search(dummy_data)
        self.assertEqual(out, [{i} for i in range(32)])
        out = [list(g) for g in dummy.batch_itersearch(dummy_data)]
        self.assertEqual(out, [[{i} for i in range(32)] for _ in range(32)])