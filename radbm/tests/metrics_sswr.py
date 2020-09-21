import unittest
import numpy as np
from radbm.metrics.sswr import HaltingChronoSSWR, HaltingCounterSSWR, ChronoSSWR, CounterSSWR

def moc(N, K, k):
    #short for matching oracle cost
    return k*(N+1)/(K+1)

class TestSSWR(unittest.TestCase):
    def test_countersswr(self):
        #test 1
        relevant = {0,1,2,3,4}
        delta_gen = (s for s in [{0,1}, {2,3}, {4,5}])
        out = CounterSSWR(relevant, delta_gen, N=20)
        
        #            find 4 in {4,5} + |{0,1,2,3}|   +     #generator calls
        expected_out = (moc(2,1,1) + len({0,1,2,3}) + len([{0,1}, {2,3}, {4,5}])) / moc(20,5,5)
        self.assertEqual(out, expected_out)
        
        #test 2 (recall=0.8) (we need 4 out of 5)
        relevant = {0,1,2,3,4}
        delta_gen = (s for s in [{0,1}, {2,3}, {4,5}])
        out = CounterSSWR(relevant, delta_gen, N=20, recall=0.8)
        
        #            find 2,3 in {2,3} + |{0,1}| + #generator calls
        expected_out = (moc(2,2,2) + len({0,1}) + len([{0,1}, {2,3}])) / moc(20,5,4)
        self.assertEqual(out, expected_out)
        
        #test 3 (with bad candidates)
        relevant = {0,1,2,3,4}
        delta_gen = (s for s in [{0,1,10,11}, {2,3,12}, {4,5,13}])
        out = CounterSSWR(relevant, delta_gen, N=20)
        
        #            find 4 in {4,5,13} + |{0,1,10,11,2,3,12}| + #generator calls
        expected_out = (moc(3,1,1) +   len({0,1,10,11,2,3,12}) + 3) / moc(20,5,5)
        self.assertEqual(out, expected_out)
        
        #test 4 (with bad candidates + duplicates (and on_duplicate_candidates='ignore'))
        relevant = {0,1,2,3,4}
        delta_gen = (s for s in [{0,1,10,11}, {2,3,12,0,10}, {4,5,13,1,2,12}])
        out = CounterSSWR(relevant, delta_gen, N=20, on_duplicate_candidates='ignore')
        
        #            find 4 in {4,5,13} + |{0,1,10,11,2,3,12}| + #generator calls
        expected_out = (moc(3,1,1) +   len({0,1,10,11,2,3,12}) + 3) / moc(20,5,5)
        self.assertEqual(out, expected_out)
        
        #test 5 (halting + bad candidates + recall=5/6)
        relevant = {0,1,2,3,4,5}
        delta_gen = (s for s in [{0,1}, {3,10}, {12}])
        out = CounterSSWR(relevant, delta_gen, N=20, recall=5/6, allow_halt=True)
        
        #        find 2 out of 3 in 15 elements + |{0,1,10,11,2,3,12}| + #generator calls
        expected_out = (moc(15,3,2) + len({0,1,3,10,12}) + 3) / moc(20,6,5)
        self.assertEqual(out, (expected_out, True))
        
    def test_haltingcountersswr(self):
        relevant = {0,1,2,3,4}
        delta_gen = (s for s in [{0,1,5}, {2,3}, {4,6,7}])
        sswrs, halts = HaltingCounterSSWR(relevant, delta_gen, N=20, max_halt=4)
        expected_sswrs = np.array([
            moc(20,5,5) + 0 + 0, #halt
            moc(17,3,3) + 3 + 1, #halt
            moc(15,1,1) + 5 + 2, #halt
            moc(3,1,1) + 5 + 3, #not halt
            moc(3,1,1) + 5 + 3, #not halt (same->copy)
        ])/moc(20,5,5)
        expected_halts = np.array([1,1,1,0,0])
        self.assertTrue(np.allclose(sswrs, expected_sswrs))
        self.assertTrue(np.allclose(halts, expected_halts))
        
        #with recall = 6/7 and max_hatl=4
        relevant = {0,1,2,3,4,5,6}
        delta_gen = (s for s in [{0,1,7}, {2,3}, {4,8,9}, {10,11}, {5}]) #<-- 5 found at step 5<4
        sswrs, halts = HaltingCounterSSWR(relevant, delta_gen, N=20, max_halt=4, recall=6/7)
        expected_sswrs = np.array([
            moc(20,7,6) + 0 + 0, #halt
            moc(17,5,4) + 3 + 1, #halt
            moc(15,3,2) + 5 + 2, #halt
            moc(12,2,1) + 8 + 3, #halt
            moc(10,2,1) + 10 + 4, #halt
        ])/moc(20,7,6)
        expected_halts = np.array([1,1,1,1,1])
        self.assertTrue(np.allclose(sswrs, expected_sswrs))
        self.assertTrue(np.allclose(halts, expected_halts))
    
    def test_chronosswr(self):
        #make sure it runs
        relevant = {0,1,2,3,4}
        delta_gen = (s for s in [{0,1}, {2,3}, {4,5}])
        out = ChronoSSWR(relevant, delta_gen, N=20)
        
    def test_haltingchronosswr(self):
        #make sure it runs
        relevant = {0,1,2,3,4}
        delta_gen = (s for s in [{0,1}, {2,3}, {4,5}])
        out = HaltingChronoSSWR(relevant, delta_gen, N=20, max_halt=4)
        
    def test_generator_halt_error(self):
        relevant = {0,1,2,3,4}
        
        #test 1 (non-empty)
        delta_gen = (s for s in [{0,1}])
        with self.assertRaises(LookupError):
            CounterSSWR(relevant, delta_gen, N=20, on_duplicate_candidates='raise')
            
        #test 2 (empty)
        delta_gen = (s for s in [])
        with self.assertRaises(LookupError):
            CounterSSWR(relevant, delta_gen, N=20, on_duplicate_candidates='raise')
    
    def test_duplicate_candidates(self):
        relevant = {0,1,2,3,4}
        delta_gen = (s for s in [{0,1,10,11}, {2,3,12,0,10}, {4,5,13,1,2,12}])
        with self.assertRaises(RuntimeError):
            CounterSSWR(relevant, delta_gen, N=20, on_duplicate_candidates='raise')
    
    def test_bad_duplicate_candidates_option(self):
        relevant = {0,1,2,3,4}
        delta_gen = (s for s in [{0,1,10,11}, {2,3,12,0,10}, {4,5,13,1,2,12}])
        with self.assertRaises(ValueError):
            CounterSSWR(relevant, delta_gen, N=20, on_duplicate_candidates='bad_option')