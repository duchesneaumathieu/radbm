import numpy as np
from scipy.special import comb
from radbm.loaders.base import IRLoader
from radbm.utils.numpy.random import (
    unique_randint,
    no_subset_unique_randint,
    uniform_n_choose_k_by_enumeration,
    uniform_n_choose_k_by_rejection,
)
from radbm.utils.numpy.logical import issubset_product_with_trie, adjacency_list_to_matrix
from radbm.utils.numpy.function import log_comb, softplusinv

class ConjunctiveBooleanRSS(IRLoader):
    ENUMERATE_MAX=int(1e6)
    
    def __init__(self, k, l, m, n, mode='balanced', which='train', backend='numpy', device='cpu', rng=np.random):
        super().__init__(mode=mode, which=which, backend=backend, device=device, rng=rng)
        self.params = int(k), int(l), int(m), int(n)
        self.k, self.l, self.m, self.n = self.params
        qterms, dterms, self.relevants = self.generate_database_stucture()
        self.register_switch('qterms', qterms)
        self.register_switch('dterms', dterms)
        getattr(self, self.which)()
    
    def get_available_modes(self):
        return {
            'balanced',
            'block',
        }
    
    def generate_subsets(self, dterms):
        """
        Parameters
        ----------
        documents : numpy.ndarray (ndim: 2, shape: (n, l))
            documents[i] is the i-th document.
        k : int
            The size of each query.
        rng : numpy.random.generator.Generator

        Returns
        -------
        queries : numpy.ndarray (ndim: 2, shape: (n, k))
            queries[i] is a subset of documents[i] sampled uniformly.
        """
        k, l, m, n = self.params
        n, l = dterms.shape #rewrite n (for batch sampling)
        d0 = np.repeat(range(n), k)
        d1 = unique_randint(0, l, n, k, rng=self.rng.rng).flatten()
        qterms = dterms[(d0, d1)].reshape(n, k)
        return qterms

    def generate_database_stucture(self):
        k, l, m, n = self.params
        if k > l:
            msg = 'We need k <= l, otherwise the number of relevants document(s) will always be zero. Got {} and {} respectively.'
            raise ValueError(msg.format(k, l))
        if n > comb(m, l):
            msg = 'Cannot generate n unique documents. We need n <= comb(m, l) however, n = {} and comb({}, {}) = {}.'
            raise ValueError(msg.format(n, m, l, comb(m, l, exact=True)))
        if comb(m, l) < ConjunctiveBooleanRSS.ENUMERATE_MAX:
            dterms = uniform_n_choose_k_by_enumeration(m, l, n, rng=self.rng.rng)
        elif n < comb(m, l)/2:
            dterms = uniform_n_choose_k_by_rejection(m, l, n, rng=self.rng.rng)
        else:
            msg = ('Cannot generate documents. comb(m, l) > enumerate_max which disable enumeration + np.random.choice technique.'
                   ' Also, n < comb(m, l)/2 which would make rejection sampling very slow. Consider increasing ENUMERATE_MAX or '
                   'decreasing n.')
            raise ValueError(msg)
        list(map(self.rng.rng.shuffle, dterms)) #shuffling the index terms of each documents (inplace)

        qterms = self.generate_subsets(dterms)
        relevants = issubset_product_with_trie(qterms, dterms)
        return qterms, dterms, relevants
    
    def residual_log_prob(self, k, l, m):
        """
        Numerically stable numpy.log((comb(m-k, l-k)-1)/(comb(m, l)-1))
        """
        return softplusinv(log_comb(m-k, l-k)) - softplusinv(log_comb(m, l))
    
    def get_relation_prob(self):
        k, l, m, n = self.params
        if comb(m, l) == 1: return 1
        if k == l: return 1/n
        #residual = (comb(m-k, l-k)-1)/(comb(m, l)-1) #unstable
        residual = np.exp(self.residual_log_prob(k, l, m))
        return 1/n + (n-1)/n*residual
    
    def get_relation_log_prob(self):
        k, l, m, n = self.params
        if comb(n, l) == 1: return 0.
        if k == l: return -np.log(n)
        log_residual = self.residual_log_prob(k, l, m)
        a = np.log(1 + (n - 1)*np.exp(log_residual))
        return a - np.log(n)
    
    def _get_nbatch(self, batch_size, n):
        nfull_batch = n//batch_size
        nbatch = nfull_batch if n % batch_size == 0 else nfull_batch + 1
        return nbatch
    
    def iter_documents(self, batch_size, maximum=np.inf, rng=np.random):
        which = self.which
        n = min(maximum, self.n)
        nbatch = self._get_nbatch(batch_size, n)
        for i in range(nbatch):
            if self.which != which:
                msg = 'self.which changed while iterating, should be "{}" but is now "{}".'
                raise RuntimeError(msg.format(which, self.which))
            start = i*batch_size
            end = min(start+batch_size, n)
            yield self.dterms.data[start:end], list(range(start, end))
    
    def iter_queries(self, batch_size, maximum=np.inf, rng=np.random):
        which = self.which
        n = min(maximum, self.n)
        nbatch = self._get_nbatch(batch_size, n)
        for i in range(nbatch):
            if self.which != which:
                msg = 'self.which changed while iterating, should be "{}" but is now "{}".'
                raise RuntimeError(msg.format(which, self.which))
            start = i*batch_size
            end = min(start+batch_size, n)
            yield self.qterms.data[start:end], self.relevants[start:end]

    def generate_balanced_batch(self, bs):
        k, l, m, n = self.params
        halfbs = bs//2
        relevants = np.zeros((bs,), dtype=bool)
        relevants[:halfbs] = 1
        dterms = unique_randint(0, m, bs, l, rng=self.rng.rng)
        qterms = np.zeros((bs, k), dtype=int)
        qterms[:halfbs] = self.generate_subsets(dterms[:halfbs])
        qterms[halfbs:] = no_subset_unique_randint(0, m, bs-halfbs, k, dterms[halfbs:], rng=self.rng.rng)
        return qterms, dterms, relevants

    def generate_block_batch(self, bs):
        k, l, m, n = self.params
        dterms = unique_randint(0, m, bs, l, rng=self.rng.rng)
        qterms = self.generate_subsets(dterms)
        relevants = issubset_product_with_trie(qterms, dterms)
        block = adjacency_list_to_matrix(relevants, bs)
        return qterms, dterms, block
    
    def batch(self, size):
        if self.mode=='balanced':
            q, d, r = self.generate_balanced_batch(size)
        elif self.mode=='block':
            q, d, r = self.generate_block_batch(size)
            
        q, d, r = map(self.dynamic_cast, [q, d, r])
        return q, d, r