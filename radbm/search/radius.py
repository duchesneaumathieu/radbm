import itertools, torch
import numpy as np
from radbm.search.base import BaseSDS

def cast_numpy(data):
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return data

class HammingRadiusSDS(BaseSDS):
    """
    A Seach Data Structure for binary vectors, it searches within a Hamming radius.
    This uses hash tables to speed up the search.
    
    Parameters
    ----------
    nbits : int
        The number of bits the binary vectors will have.
    radius : int
        The Hamming radius to search inside. Should be small as the masks are precomputed.
        The memory taken is nbits * sum_{i=0}^radius comb(nbits, i).
    ntables : int
        The number of Hash tables to uses for the documents.
    """
    def __init__(self, nbits, radius, ntables=1):
        self.nbits = nbits
        self.radius = radius
        self.ntables = ntables
        comb = sum(
            (list(itertools.combinations(range(nbits),i)) for i in range(radius+1)),
            []
        )
        rows, cols = zip(*[(i,j) for i, c in enumerate(comb) for j in c])
        self.masks = np.zeros((len(comb), nbits), dtype=np.bool)
        self.masks[(rows, cols)] = True
        self.reset()
        
    def reset(self):
        """
        Empty the hash tables
        
        Returns
        -------
        self
        """
        self.tables = [dict() for _ in range(self.ntables)]
        return self
        
    def search(self, query):
        query = cast_numpy(query)
        indexes = set()
        for h in query^self.masks:
            th = tuple(bool(b) for b in h)
            for table in self.tables:
                if th in table:
                    indexes.update(table[th])
        return indexes
    
    def insert(self, document, index):
        document = cast_numpy(document)
        hashes = document^self.masks[:self.ntables]
        for h, table in zip(hashes, self.tables):
            th = tuple(bool(b) for b in h)
            if th in table: table[th].add(index)
            else: table[th] = {index}
        return self
    
    def itersearch(self, query, yield_empty=False):
        """
        Generator that search in the tables with a query in the form of a 
        Multi-Bernoulli distribution parametrized in log probabilities.
        This will search in each tables with each of the top (nlookups)
        outcomes. Everytime a set of indexes is found, this generator will
        yield it. If yield_empty is True, empty set will also be yield.

        Parameters
        ----------
        log_probs : numpy.ndarray (ndim == 1 or 2)
            If ndim==1: The Multi-Bernoulli distribution parametrized in log probabilities
            If ndim==2: len(log_probs) should be 2. The first element should be log
            probabilities that bits are zero and the second element should be the log
            probabilities that bits are one. This is for numerical stability
            (i.e. when probability are too close to 1)
        yield_empty : bool (optinal)
            Whether empty set should be yielded (default False)

        Yields
        ------
        indexes : set
            The newly found indexes
        """
        query = cast_numpy(query)
        yielded = set()
        for bits in query^self.masks:
            bits = tuple(bool(b) for b in bits)
            for table in self.tables:
                if bits in table:
                    indexes = table[bits]
                    new_indexes = indexes - yielded
                    yielded.update(new_indexes)
                    if yield_empty or new_indexes:
                        yield new_indexes
                elif yield_empty:
                    yield set()