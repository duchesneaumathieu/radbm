import numpy as np
from radbm.utils.generators import poisson_binomial_top_k_generator
from radbm.searches.base import Search

class PoissonBinomialHashing(Search):
    def __init__(self, ntables, nlookups):
        self.ntables = ntables
        self.nlookups = nlookups
        self.tables = [dict() for _ in range(ntables)]
        
    def get_buckets_avg_size(self):
        return [np.mean(list(map(len, t.values()))) for t in self.tables]
    
    def get_buckets_max_size(self):
        return [np.max(list(map(len, t.values()))) if t else 'nan' for t in self.tables]
        
    def __repr__(self):
        r = 'Search: {}\nTables size: {}\nBuckets avg size: {}\nBuckets max size: {}'
        return r.format(
            self.__class__.__name__,
            ', '.join(map(str, map(len, self.tables))),
            ', '.join(['{:.2f}'.format(s) for s in self.get_buckets_avg_size()]),
            ', '.join(map(str, self.get_buckets_max_size())),
        )
        
    def insert(self, log_probs, i):
        gen = poisson_binomial_top_k_generator(log_probs, k=self.ntables)
        for n, bits in enumerate(gen):
            table = self.tables[n]
            if bits in table: table[bits].add(i)
            else: table[bits] = {i}
                
    def search(self, log_probs):
        ids = set()
        for bits in poisson_binomial_top_k_generator(log_probs, k=self.nlookups):
            for table in self.tables:
                if bits in table:
                    ids = ids.union(table[bits])
        return ids
    
    def get_state(self): return self.tables
    def set_state(self, state): self.tables=state; return self