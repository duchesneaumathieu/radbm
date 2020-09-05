import unittest
from radbm.metrics.sswr import ChronoSSWR, CounterSSWR

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
    
    def test_chronosswr(self):
        #make sure it runs
        relevant = {0,1,2,3,4}
        delta_gen = (s for s in [{0,1}, {2,3}, {4,5}])
        out = ChronoSSWR(relevant, delta_gen, N=20)
        
    def test_generator_exited_early(self):
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