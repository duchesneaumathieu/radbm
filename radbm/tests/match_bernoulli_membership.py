from radbm.match.bernoulli import MultiBernoulliMembershipMatch
import unittest, functools, itertools, torch
import numpy as np

def mbp(pi, B):
    return pi[B].prod() * (1-pi[~B]).prod()

def prod(*args):
    return functools.reduce(lambda a, b: a*b, args)

def exact_membership_match(q, d):
    #sum over x outcomes
    l, n = len(d), len(q)
    p = 0
    for bits in itertools.product(*n*[(False, True)]):
        bits = torch.tensor(bits)
        p += mbp(q, bits) * prod(*(1-mbp(di, bits) for di in d))
    return 1-p

def approx_membership_match(q, d, N, rng=np.random):
    #Monte Carlo
    q = q.numpy()
    d = d.numpy()
    l, n = len(d), len(q)
    Bq = torch.tensor(rng.binomial(1, q[None], size=(N, 1, n)), dtype=bool)
    Bd = torch.tensor(rng.binomial(1, d[None], size=(N, l, n)), dtype=bool)
    match = (~(Bq ^ Bd)).all(dim=2).any(dim=1)
    return match.float().mean()

class TestMultiBernoulliMembershipMatch(unittest.TestCase):
    def test_multi_bernoulli_membership_match(self):
        bs, l, n = 8, 4, 5
        rng = np.random.RandomState(0xcafe)
        np_q_logit = rng.uniform(-9, 9, (bs, n))
        np_d_logit = rng.uniform(-9, 9, (bs, l, n))
        q_logit = torch.tensor(np_q_logit, dtype=torch.float32)
        d_logit = torch.tensor(np_d_logit, dtype=torch.float32)
        q = torch.sigmoid(q_logit)
        d = torch.sigmoid(d_logit)
        
        match = MultiBernoulliMembershipMatch()
        log_p0, log_p1 = match.soft_match(q_logit, d_logit)
        self.assertTrue(torch.allclose(log_p0.exp() + log_p1.exp(), torch.ones_like(log_p0)))
        
        p = log_p1.exp()
        for i in range(bs):
            expected_p = exact_membership_match(q[i], d[i])
            self.assertAlmostEqual(float(p[i]), float(expected_p), places=6)
            
            expected_p = approx_membership_match(q[i], d[i], N=100000, rng=rng)
            self.assertAlmostEqual(float(p[i]), float(expected_p), places=2)
            
        match = MultiBernoulliMembershipMatch(terms=[1])
        log_p0, log_p1 = match.soft_match(q_logit, d_logit) #make sure it runs
            
    def test_multi_bernoulli_membership_match_error(self):
        bs, l, n = 8, 4, 5
        rng = np.random.RandomState(0xcafe)
        np_q_logit = rng.uniform(-9, 9, (bs, n))
        np_d_logit = rng.uniform(-9, 9, (bs, l, n))
        q_logit = torch.tensor(np_q_logit, dtype=torch.float32)
        d_logit = torch.tensor(np_d_logit, dtype=torch.float32)
        
        match = MultiBernoulliMembershipMatch()
        match.soft_match(q_logit, d_logit) #make sure this works
        #If x.ndim != 2.
        with self.assertRaises(ValueError):
            match.soft_match(q_logit[0], d_logit)
            
        #If y.ndim != 3.
        with self.assertRaises(ValueError):
            match.soft_match(q_logit, d_logit[0])
            
        #If x.shape[0] != y.shape[0] (i.e., different batch size).
        with self.assertRaises(ValueError):
            match.soft_match(q_logit[:4], d_logit)
        with self.assertRaises(ValueError):
            match.soft_match(q_logit, d_logit[:4])
            
        #If x.shape[1] != y.shape[2] (i.e., different number of bits).
        with self.assertRaises(ValueError):
            match.soft_match(q_logit[:, :4], d_logit)
        with self.assertRaises(ValueError):
            match.soft_match(q_logit, d_logit[:, :, :4])
            
        #If no valid odd term (i.e., within 1 up to l) is given in terms.
        match = MultiBernoulliMembershipMatch(terms=[2,4])
        with self.assertRaises(ValueError):
            match.soft_match(q_logit, d_logit)
            
    def test_multi_bernoulli_membership_hard_match(self):
        q_logit = torch.tensor([
            [+.5, -.5, -.5, +.5],
            [+.5, -.5, -.5, +.5]
        ], dtype=torch.float32)
        d_logit = torch.tensor([
            [
                [+.5, -.5, -.5, +.5], #match
                [-.5, -.5, -.5, -.5], #not match
            ], #match
            [
                [+.5, +.5, -.5, +.5], #not match
                [+.5, -.5, -.5, -.5], #not match
            ], #not match
        ], dtype=torch.float32)
        expected_hard_match = torch.tensor([True, False])
        match = MultiBernoulliMembershipMatch()
        hard_match = match.hard_match(q_logit, d_logit)
        self.assertTrue(torch.equal(hard_match, expected_hard_match))