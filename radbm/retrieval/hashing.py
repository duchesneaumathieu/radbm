import numpy as np
from radbm.retrieval.base import Retrieval
from radbm.utils.stats.generators import multi_bernoulli_top_k_generator


class MultiBernoulliHashTables(Retrieval):
    """
    This algorithm build N hash tables from documents in the form
    of Multi-Bernoulli distributions. From a documents (i.e. a
    distribution) we find the i-th most probable outcome (which is
    a binary vector) and use it as a key for the corresponding
    i-th hash table.
    
    For retrieval, a query (i.e. a Multi-Bernoulli distribution) 
    we use the M most probable outcomes (which are binary vectors)
    to do M lookups in the hash tables. In this context, a lookups
    means looking in each tables. In total NxM hash table calls
    will be made.
    
    Parameters
    ----------
    ntables : int
        The number of tables to build
    nlookups : int, optional (default=1)
        The number of lookups to do when searching
    
    Notes
    -----
        The Multi-Bernoulli distribution are always parametrized with the
        vector of log probabilities for the bits to be one.
    """
    def __init__(self, ntables, nlookups=1):
        self.ntables = ntables
        self.nlookups = nlookups
        self.tables = [dict() for _ in range(ntables)]
        
    def get_buckets_avg_size(self):
        """
        Returns
        -------
        buckets_avg_size : list of float
            The average number of documents per buckets for each
            hash tables
        """
        return [np.mean(list(map(len, t.values()))) for t in self.tables]
    
    def get_buckets_max_size(self):
        """
        Returns
        -------
        buckets_max_size : list of float
            The maximum number of documents per buckets for each
            hash tables
        """
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
        """
        Insert a unique document's index in the each tables. The document
        most be a Multi-Bernoulli distribution parametrized in log
        probabilities.
        
        Parameters
        ----------
        log_probs : numpy.ndarray
            The Multi-Bernoulli distribution parametrized in log probabilities
        i : hashable (e.g. int or tuple)
            The index of the document
        """
        gen = multi_bernoulli_top_k_generator(log_probs, k=self.ntables)
        for n, bits in enumerate(gen):
            table = self.tables[n]
            if bits in table: table[bits].add(i)
            else: table[bits] = {i}
                
    def search(self, log_probs, nlookups=None):
        """
        Search in the tables with a query in the form of a Multi-Bernoulli
        distribution parametrized in log probabilities. This will search in
        each tables with each of the top (nlookups) outcomes.
        
        Parameters
        ----------
        log_probs : numpy.ndarray
            The Multi-Bernoulli distribution parametrized in log probabilities.
        nlookups : int, optional
            The number of top outcome to uses for searching. If not specified
            the default self.nlookups is used.
            
        Returns
        -------
        indexes : set
            The indexes of each documents found in the search
        """
        indexes = set()
        nlookups = self.nlookups if nlookups is None else nlookups
        #loop for nlookups*self.ntables minus the number of empty lookups
        for new_indexes in self.batch_itersearch(log_probs, nlookups):
            indexes.update(new_indexes)
        return indexes
    
    def batch_itersearch(self, log_probs, nlookups=None):
        """
        Generator that search in the tables with a query in the form of a 
        Multi-Bernoulli distribution parametrized in log probabilities.
        This will search in each tables with each of the top (nlookups)
        outcomes. Everytime a set of indexes is found, this generator will
        yield it.

        Parameters
        ----------
        log_probs : np.ndarray
            The Multi-Bernoulli distribution parametrized in log probabilities.
        nlookups : int, optional
            The upper limit to generate the next most probable outcomes. Not to
            be confused with the number of item generated. By default, generates
            every outcomes. 

        Yields
        ------
        indexes : set
            The newly found indexes
        """
        for bits in multi_bernoulli_top_k_generator(log_probs, k=nlookups):
            for table in self.tables:
                if bits in table:
                    yield table[bits]
    
    def get_state(self):
        """
        Returns
        -------
        tables : list of dict
            The current hash tables
        """
        return self.tables
    
    def set_state(self, state):
        """
        Parameters
        -------
        state : list of dict
            The hash tables to use
        """
        self.tables = state
        return self