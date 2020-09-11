import unittest, torch
from radbm.metrics.hamming import hamming_pr_curve

class TestHammingPRCurve(unittest.TestCase):
    def test_hamming_pr_curve(self):
        queries = torch.tensor([
            [0,0,0,0],
            [0,0,1,1],
            [1,1,1,1],
        ])
        documents = torch.tensor([
            [0,0,0,1],
            [0,1,0,1],
            [1,0,1,1],
            [0,0,1,0],
        ])
        #hamming dists matrix
        #1, 2, 3, 1
        #1, 2, 1, 1
        #3, 2, 1, 3
        rel = [
            {0, 3}, #d is 1 1
            {1, 2}, #d is 2 1
            {0}, #d is 3
        ]
        #at 0 -> zero found -> p=nan, r=0
        #at 1 -> 6 found with 3 good out of 5 -> p=3/6=1/2, r=3/5
        #at 2 -> 9 found with 4 good out of 5 -> p=4/9, r=4/5
        #at 3 -> 12 found with 5 good out of 5 -> p=5/12, r=5/5=1
        #at 4 -> idem -> p=5/12, r=1
        expected_precisions = torch.tensor([float('nan'), 1/2, 4/9, 5/12, 5/12])
        expected_recalls = torch.tensor([0, 3/5, 4/5, 1, 1])
        precisions, recalls = hamming_pr_curve(documents, queries, rel)
        self.assertTrue(torch.isnan(precisions[0]))
        self.assertTrue(torch.allclose(precisions[1:], expected_precisions[1:]))
        self.assertTrue(torch.allclose(recalls, expected_recalls))
        
        #same thing but with return_valid_dists=True
        expected_dists = torch.tensor([1,2,3,4]) #not 0 because precision is undefined
        dists, precisions, recalls = hamming_pr_curve(documents, queries, rel, return_valid_dists=True)
        self.assertTrue(torch.allclose(dists, expected_dists))
        self.assertTrue(torch.allclose(precisions, expected_precisions[1:]))
        self.assertTrue(torch.allclose(recalls, expected_recalls[1:]))
        
        #error checks
        with self.assertRaises(ValueError):
            #code not same length
            hamming_pr_curve(documents[:,:-1], queries, rel, return_valid_dists=True)
        with self.assertRaises(ValueError):
            #not enough relevances
            hamming_pr_curve(documents, queries, rel[:-1], return_valid_dists=True)