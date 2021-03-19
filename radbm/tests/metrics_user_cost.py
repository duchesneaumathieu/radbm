import unittest, torch
from radbm.metrics.hamming import hamming_distance
from radbm.metrics import (
    user_cost_at_k_original,
    user_cost_at_k_from_counts,
    user_cost_at_k,
    user_cost_at_k_from_scores,
)
    

class TestUCK(unittest.TestCase):
    def test_user_cost_at_k_small(self):
        scores = torch.tensor([60, 20, 70, 40, 50, 50, 30, 70, 50, 40]) #the score of each document
        #       documents'id:   0,  1,  2,  3,  4,  5,  6,  7,  8,  9
        #thus the total pre-order is: 1 < 6 < 3,9 < 4,5,8 <  0 < 2,7
        #                         t = 0   1    2      3      4    5
        #i.e. the set of candidates are:
        candidates = [{1}, {6}, {3, 9}, {4, 5, 8}, {0}, {2, 7}]
        relevants = torch.tensor([8, 3, 5]) #the indices of the relevant documents
        #their scores are 50, 40, 50 respectivaly
        
        #One relevant documents will be found at t=2 (3). The unrelevant 1 and 6 will be retrieved before.
        #It is possible possible that 9 is retrieved before 3 aswell.
        #The UC1 is thus |{1, 6}| + C(|{3, 9}|, |{3}|, 1-|{}|) = 2 + 1*(2+1)/(1+1) = 2 + 3/2 = 7/2
        
        #Two relevant documents will be found at t=3 (3 and (5 or 8)). WLOG assume it is 8.
        #Then, {1, 6, 3, 9} will be retrieved first and {4,5} might be retrieved first.
        #The UC2 is thus |{1, 6, 3, 9}| + C(|{4, 5, 8}|, |{5,8}|, 2-|{3}|) = 4 + 1*(3+1)/(2+1) = 4 + 4/3 = 16/3
        
        #Three relevant documents will be found at t=3 (3, 5 and 8).
        #Then, {1, 6, 3, 9} will be retrived first and {4} might be retrieved first.
        #The UC3 is thus |{1, 6, 3, 9}| + C(|{4, 5, 8}|, |{5, 8}|, 3-|{3}|) = 4 + 2*(3+1)/(2+1) = 4 + 8/3 = 20/3
        
        #finally,
        expected_uck = torch.tensor([7/2, 16/3, 20/3])
        relevants_set = set(relevants.tolist()) #i.e. {8, 3, 5}
        
        #original algorithm
        uck = user_cost_at_k_original(candidates, relevants_set)
        self.assertTrue(torch.allclose(expected_uck, uck))
        
        #vectorized implementation
        uck = user_cost_at_k(candidates, relevants_set)
        self.assertTrue(torch.allclose(expected_uck, uck))
        
        #computing from the scoring directly (more vectorization)
        uck = user_cost_at_k_from_scores(scores, relevants)
        self.assertTrue(torch.allclose(expected_uck, uck))
        
    def test_user_cost_at_k_big(self):
        #make sure the three implementation gives the same results
        #even if we don't know the true uck
        
        queries = torch.randint(0, 2, (1, 64), dtype=torch.bool)
        documents = torch.randint(0, 2, (10000, 64), dtype=torch.bool)
        bool_relevants = torch.randint(0, 2, (10000,), dtype=torch.bool)
        int64_relevants = torch.where(bool_relevants)[0]
        set_relevants = set(int64_relevants.tolist())
        scores = hamming_distance(queries, documents, dim=-1)
        
        #bad but simple way of computing the set of candidates from score
        unique_scores = torch.unique(scores)
        candidates = [set(torch.where(scores==i)[0].tolist()) for i in unique_scores]
        
        uck_original = user_cost_at_k_original(candidates, set_relevants)
        uck_vectorized = user_cost_at_k(candidates, set_relevants)
        uck_scores_bool = user_cost_at_k_from_scores(scores, bool_relevants)
        uck_scores_int64 = user_cost_at_k_from_scores(scores, int64_relevants)
        
        self.assertTrue(torch.allclose(uck_original, uck_vectorized))
        #self.assertTrue(torch.allclose(uck_vectorized, uck_scores_bool))
        self.assertTrue(torch.allclose(uck_scores_bool, uck_scores_int64))
        
    def test_duplicate_relevants(self):
        #this cannot happens for user_cost_at_k and user_cost_at_k_original
        queries = torch.randint(0, 2, (1, 64), dtype=torch.bool)
        documents = torch.randint(0, 2, (10000, 64), dtype=torch.bool)
        bool_relevants = torch.randint(0, 2, (10000,), dtype=torch.bool)
        int64_relevants_unique = torch.where(bool_relevants)[0]
        int64_relevants_double = torch.cat(2*[int64_relevants_unique])
        scores = hamming_distance(queries, documents, dim=-1)
        
        uck_scores_unique = user_cost_at_k_from_scores(scores, int64_relevants_unique)
        uck_scores_double = user_cost_at_k_from_scores(scores, int64_relevants_double)
        self.assertTrue(torch.allclose(uck_scores_unique, uck_scores_double))
    
    def test_errors(self):
        queries = torch.randint(0, 2, (1, 64), dtype=torch.bool)
        documents = torch.randint(0, 2, (10000, 64), dtype=torch.bool)
        bool_relevants = torch.randint(0, 2, (10000,), dtype=torch.bool)
        int64_relevants = torch.where(bool_relevants)[0]
        set_relevants = set(int64_relevants.tolist())
        scores = hamming_distance(queries, documents, dim=-1)
        
        #bad but simple way of computing the set of candidates from score
        unique_scores = torch.unique(scores)
        candidates = [set(torch.where(scores==i)[0].tolist()) for i in unique_scores]
        
        bad_size_bool_relevants = bool_relevants[:9999]
        with self.assertRaises(ValueError):
            user_cost_at_k_from_scores(scores, bad_size_bool_relevants)
        
        bad_dtype_relevants = int64_relevants.float()
        with self.assertRaises(TypeError):
            user_cost_at_k_from_scores(scores, bad_dtype_relevants)
        
        to_much_set_relevants = set_relevants.add(10000) #10000 does not exists.
        with self.assertRaises(ValueError):
            user_cost_at_k_original(candidates, set_relevants)
            
        with self.assertRaises(ValueError):
            user_cost_at_k(candidates, set_relevants)
            
        candidates_size = torch.tensor([len(c) for c in candidates])
        relevant_counts = torch.tensor([len(set_relevants.intersection(c)) for c in candidates])
        user_cost_at_k_from_counts(candidates_size, relevant_counts) #should work
        not_enough_relevant_counts = relevant_counts[:-1]
        with self.assertRaises(ValueError):
            user_cost_at_k_from_counts(candidates_size, not_enough_relevant_counts)