import unittest
import numpy as np
from radbm.search import KeyValueHeap

class TestKeyValueHeap(unittest.TestCase):
    def test_keyvalue_heapsort(self):
        values = np.random.randint(0, 1000, 10000) #many collisions
        keys = np.random.normal(0, 1, 10000) #no collisions
        heap = KeyValueHeap(*zip(keys, values), key=lambda item: item[0])
        
        #make sure heap.batch_insert works like initialization insert
        heap2 = KeyValueHeap()
        heap2.batch_insert(zip(keys, values), key=lambda item: item[0])
        self.assertEqual(heap, heap2)
        
        #test set/get_state
        heap3 = KeyValueHeap(1, 2, 3, 4, 5, return_counts=True)
        heap3.set_state(heap.get_state())
        self.assertEqual(heap, heap3)
        
        
        #test search
        minitem = heap.search()
        self.assertEqual(minitem, min(zip(keys, values), key=heap.get_key))
        
        #assert batch_search raises
        with self.assertRaises(NotImplementedError):
            heap.batch_search()
            
        #test sort
        heap_keys = list(heap.keys())
        self.assertNotEqual(sorted(heap_keys), heap_keys) #it is highly improbable that the keys in heap order are sorted.
        heap.sort()
        sort_keys = list(heap.keys())
        self.assertEqual(sorted(sort_keys), sort_keys) #now they should be sorted.
        
    def test_keyvalue_counts(self):
        """           5
                8           9
            10     12   11    17
          11  12 23  
        """
        heap = KeyValueHeap(5, 8, 9, 10, 12, 11, 17, 11, 12, 23)
        s, c = heap.insert(6) #swap with 12 and 8, comp with 12, 8, and 5 (2 swap and 3 comp)
        """           5
                6           9
            10     8    11    17
          11  12 23  12
        """
        expected_heap = [5, 6, 9, 10, 8, 11, 17, 11, 12, 23, 12]
        self.assertEqual((expected_heap, 2, 3), (heap, s, c))
        v, s, c = heap.pop(return_counts=True)
        #move 12 to root: compare 6 and 9, compare 12 with 6, swap 6 with 12, compare 10 and 8,
        #compare 8 and 12, swap 12 and 8, compare 12 and 23, stop (2 swaps and 5 comp)
        """           6
                8           9
            10     12    11    17
          11  12 23  
        """
        expected_heap = [6, 8, 9, 10, 12, 11, 17, 11, 12, 23]
        self.assertEqual((expected_heap, 2, 5), (heap, s, c))