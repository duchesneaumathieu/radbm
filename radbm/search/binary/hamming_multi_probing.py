import torch
import numpy as np
from itertools import chain, combinations
from radbm.search.base import BaseSDS, Itersearch
from radbm.search import DictionarySearch
from radbm.utils.torch import tuple_cast, numpy_cast

def _check_dtype_is_bool(x):
    if x.dtype not in (np.bool, torch.bool):
        msg = f'dtype must be bool, got {x.dtype}'
        raise TypeError(msg)
            
def _iter_errors(radius, n):
    #yield [], [0], [1], ..., [n], [0,1], [0,2], ..., [0,n], ..., [1,2], [1,3], ..., [n-1, n], [1,2,3], ...
    return map(list, chain.from_iterable(combinations(range(n), r) for r in range(radius+1)))
        
class HammingMultiProbing(BaseSDS):
    r"""
    The binary version of the multi-probing algorithm `Modeling lsh for performance tuning <https://dl.acm.org/doi/abs/10.1145/1458082.1458172>`__.
    
    Parameters
    ----------
    insert_radius : int (optional)
        The Hamming distance radius for insertion. The ball of radius insert_radius around each document will map to
        the document's index in the hash table. Can be overwritten in the batch_insert and insert methods using the radius
        keyword. (default: 0)
    search_radius : int (optional)
        The Hamming distance radius for searching. The ball of radius search_radius around each query will be looked up
        in the hash table. Can be overwritten in the batch_search and search methods using the radius
        keyword. (default: 0)
    probing : str (optional)
        Should be 'all' or 'align'. If 'all', search will retrieve all documents matching with one of the query's tag. If
        'align', search will retrieve document s.t. at least one of their i-th tag matchs with the i-th tag of the query. 
        (default: 'all')
    halt_cost : float (optional)
        The cost at which to halt when generating candidates in the itersearch and batch_itersearch methods. The value
        can be overwritten in those methods using the halt_cost keyword. (default: np.inf)
    """
    def __init__(self, insert_radius=0, search_radius=0, probing='all', halt_cost=np.inf):
        if probing not in ('all', 'align'):
            raise ValueError(f'probing must be "all" or "align", got {probing}')
        self.insert_radius = insert_radius
        self.search_radius = search_radius
        self.probing = probing
        self.halt_cost = halt_cost
        self.table = DictionarySearch()
        
    def batch_insert(self, documents, indexes, radius=None):
        r"""
        Parameters
        ----------
        documents : numpy.ndarray or torch.tensor (dtype: bool, ndim: 2 or 3)
            The documents, must be binary vectors.
        indexes : iterable of hashable
            A sequence of unique identifier for each corresponding document.
        radius : None or int (optional)
            The radius at which to insert. If None, self.insert_radius will be used. (default: None)
            
        Returns
        -------
        self : HammingMultiProbing
        
        Raises
        ------
        TypeError
            If documents.dtype is not boolean.
        """
        _check_dtype_is_bool(documents)
        radius = self.insert_radius if radius is None else radius
        if documents.ndim == 2:
            l = 1
            bs, n = documents.shape
        elif documents.ndim == 3:
            bs, l, n = documents.shape
            documents = documents.view(bs*l, n)
        else:
            raise ValueError(f'documents.ndim must be 2 or 3, got {documents.ndim} (shape = {documents.shape})')
        indexes = list(x for y in indexes for x in l*[y]) #repeat each index l time
        for err in _iter_errors(radius, n):
            documents[:,err] ^= True #fast inplace on GPU
            keys = tuple_cast(documents)
            documents[:,err] ^= True #reversing the inplace operation
            keys = keys if self.probing=='all' else zip(bs*list(range(l)), keys) #append the tag id to each tag
            self.table.batch_insert(keys, indexes)
        return self
    
    def batch_search(self, queries, radius=None):
        r"""
        Parameters
        ----------
        queries : numpy.ndarray or torch.tensor (dtype: bool, ndim: 2 or 3)
            The queries, must be binary vectors.
        radius : None or int (optional)
            The radius at which to search. If None, self.search_radius will be used. (default: None)
            
        Returns
        -------
        indexes : list if sets
            Each set contains the retrieved documents' identifier of the corresponding query.
        
        Raises
        ------
        TypeError
            If queries.dtype is not boolean.
        """
        _check_dtype_is_bool(queries)
        radius = self.search_radius if radius is None else radius
        if queries.ndim == 2:
            k = 1
            bs, n = queries.shape
        elif queries.ndim == 3:
            bs, k, n = queries.shape
            queries = queries.view(bs*k, n)
        else:
            raise ValueError(f'queries.ndim must be 2 or 3, got {queries.ndim} (shape = {queries.shape})')
        indexes = [set() for _ in range(bs)]
        for err in _iter_errors(radius, n):
            queries[:,err] ^= True #fast inplace on GPU
            keys = tuple_cast(queries)
            queries[:,err] ^= True #reversing the inplace operation
            keys = keys if self.probing=='all' else zip(bs*list(range(k)), keys) #append the tag id to each tag
            for i, new in enumerate(self.table.batch_search(keys)):
                indexes[i//k].update(new)
        return indexes
    
    def _itersearch(self, itersearch, query, halt_cost=None, yield_cost=False, yield_empty=False, yield_duplicates=False):
        #itersearch.cost exits! (and start at zero)
        old = set()
        k, n = query.shape
        for err in _iter_errors(n, n):
            q = query.copy()
            q[:,err] ^= True
            keys = tuple_cast(q) if self.probing=='all' else zip(range(k), tuple_cast(q))
            for key in keys:
                new = self.table.search(key); itersearch.cost += 1
                if not yield_duplicates:
                    new = new - old
                    old.update(new)
                if new or yield_empty:
                    if yield_cost: yield new, itersearch.cost
                    else: yield new
                if itersearch.cost >= halt_cost:
                    return
    
    def itersearch(self, query, halt_cost=None, yield_cost=False, yield_empty=False, yield_duplicates=False):
        r"""
        Parameters
        ----------
        query : numpy.ndarray or torch.tensor (dtype: bool, ndim: 1 or 2)
            The query, must be a binary vector.
        halt_cost : None or float (optional)
            The cost at which to halt. If None, self.halt_cost will be used. (default: None)
        yield_cost : bool (optional)
            Whether to give the total cost each time a set of candidates is yielded. (default: False)
        yield_empty : bool (optional)
            Whether to yield an empty set when there is a table miss (i.e., when multi-probing tries to lookup
            for a binary vector not present in the database). (default: False)
        yield_duplicates : bool (optional)
            Whether to yield documents' identifier each time they are found. Even if they were found before.
            (default: False)
            
        Yields
        -------
        indexes : set (if yield_cost is False (default))
             A set containing the retrieved documents' identifier of the corresponding query.
        indexes, cost : set, float (if yield_cost is True)
            If yield_cost is True, tuples will be yieled where costs is the total cost of the search so far.
            
        Raises
        ------
        TypeError
            If queries.dtype is not boolean.
        """
        _check_dtype_is_bool(query)
        query = numpy_cast(query)
        query = query[None] if query.ndim==1 else query
        halt_cost = self.halt_cost if halt_cost is None else halt_cost
        return Itersearch(
            self._itersearch,
            query,
            halt_cost=halt_cost,
            yield_cost=yield_cost,
            yield_empty=yield_empty,
            yield_duplicates=yield_duplicates,
        )
    
    def __repr__(self):
        buckets_size = [len(buckets) for buckets in self.table.values()]
        return (
                f'Search: {self.__class__.__name__}\n'
                f'Table size: {len(self.table)}\n'
                f'Buckets avg size: {np.mean(buckets_size) if buckets_size else np.nan:.2f}\n'
                f'Buckets max size: {np.max(buckets_size) if buckets_size else np.nan}\n'
            )
    
    def get_state(self):
        """
        Returns
        -------
        tables : dict
            The current hash table
        """
        return {
            'insert_radius': self.insert_radius,
            'search_radius': self.search_radius,
            'halt_cost': self.halt_cost,
            'probing': self.probing,
            'table': self.table.get_state(),
        }
    
    def set_state(self, state):
        """
        Parameters
        -------
        state : dict
            The hash table to use
        """
        self.insert_radius = state['insert_radius']
        self.search_radius = state['search_radius']
        self.halt_cost = state['halt_cost']
        self.probing = state['probing']
        self.table.set_state(state['table'])
        return self
    
    def clear(self):
        """
        Clear the hash table
        """
        self.table.clear()
        return self