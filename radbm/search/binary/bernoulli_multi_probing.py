import torch, itertools
import numpy as np
from itertools import islice
from radbm.search.base import BaseSDS, Itersearch
from radbm.search import DictionarySearch
from radbm.utils.torch import tuple_cast
from radbm.utils.numpy.function import numpy_log_sigmoid
from radbm.utils.generators import likeliest_multi_bernoulli_outcomes, sorted_merge

torch_log_sigmoid = torch.nn.LogSigmoid()
torch_floats = {torch.float16, torch.float32, torch.float64}

class BernoulliMultiProbing(BaseSDS):
    r"""
    The a variant of the multi-probing algorithm `Modeling lsh for performance tuning
    <https://dl.acm.org/doi/abs/10.1145/1458082.1458172>`__ where each the the documents are
    Multi-Bernoulli distrutions and what is inserted in the hash table is the most probable
    outcomes. Similarly, the queries are Multi-Bernoulli and the hash table search is performed
    using the most probable outcomes.    
    
    Parameters
    ----------
    insert_number : int (optional)
        The number of outcomes to insert. Each document's outcomes will map to the document's index
        in the hash table. Can be overwritten in the batch_insert and insert methods using the number
        keyword. (default: 1)
    search_number : int (optional)
        The number of outcomes to search for in the hash table. The query's search_number most probable
        outcomes will be looked up in the hash table. Can be overwritten in the batch_search and search 
        methods using the number keyword. (default: 1)
    insert_alternate_tags : bool (optional)
        Whether to take outcomes from each Multi-Bernoulli in rotation, discarding the probability of
        the outcomes. This only affect the batch_insert and insert methods and can be overwritten when calling
        them. (default: True)
    search_alternate_tags : bool (optional)
        Whether to take outcomes from each Multi-Bernoulli in rotation, discarding the probability of
        the outcomes. This only affect the batch_search and search methods and can be overwritten when calling
        them. (default: False)
    probing : str (optional)
        Should be 'all' or 'align'. If 'all', search will retrieve all documents matching with one of the query's tag. If
        'align', search will retrieve document s.t. at least one of their i-th tag matchs with the i-th tag of the query. 
        (default: 'all')
    halt_cost : float (optional)
        The cost at which to halt when generating candidates in the itersearch and batch_itersearch methods. The value
        can be overwritten in those methods using the halt_cost keyword.
    cost : function (optional) (int swap, int comp, int size) -> float cost
        The cost function tells how the itertsearch show compute the engine cost. Swap, comp and size are all statistics
        from the heap.pop() and heap.push() methods used to compute the outcomes in decreasing order of probabilities.
        Swap is the number of heap swap, comp is the number of heap comparison, and size is the current heap size.
        (default: lambda swap,comp,size: swap+comp+1)
    """
    def __init__(
        self, insert_number=1, search_number=1, 
        insert_alternate_tags=True, search_alternate_tags=False, probing='all',
        halt_cost=np.inf, cost=lambda swap,comp,size: swap+comp+1):
    
        if probing not in ('all', 'align'):
            raise ValueError(f'probing must be "all" or "align", got {probing}')
        self.insert_number = insert_number
        self.search_number = search_number
        self.insert_alternate_tags = insert_alternate_tags
        self.search_alternate_tags = search_alternate_tags
        self.probing = probing
        self.halt_cost = halt_cost
        self.cost = cost
        self.table = DictionarySearch()
        
    def _parse_input(self, x):
        if not isinstance(x, (torch.Tensor, np.ndarray)):
            raise TypeError(f'Expected torch.Tensor or numpy.ndarray, got {str(type(x))}.')
            
        #adding the many Multi-Bernoulli dimension if not there.
        if x.ndim == 2:
            x = x[:, None, :]
        
        if x.ndim != 3:
            raise ValueError(f'Expected x.ndim in [0, 1], got x.ndim={x.ndim}.')
            
        #checking dtype and computing log_probs
        if isinstance(x, torch.Tensor):
            if x.dtype not in torch_floats:
                raise TypeError(f'Expected dtype to be float, got {x.dtype}.')
            log_probs0 = torch_log_sigmoid(-x.detach()).cpu().numpy()
            log_probs1 = torch_log_sigmoid(x.detach()).cpu().numpy()
        else: #x is np.ndarray
            if not issubclass(x.dtype.type, np.floating):
                raise TypeError(f'Expected dtype to be float, got {x.dtype}.')
            log_probs0 = numpy_log_sigmoid(-x)
            log_probs1 = numpy_log_sigmoid(x)
            
        return log_probs0, log_probs1
    
    def _iter_key(self, log_probs0, log_probs1, alternate_tags):
        tag_its = [
            likeliest_multi_bernoulli_outcomes(lp0, lp1, yield_log_probs=True, yield_stats=True)
            for lp0, lp1 in zip(log_probs0, log_probs1)
        ]
        if alternate_tags:
            it = zip(itertools.cycle(range(len(tag_its))), itertools.chain.from_iterable(zip(*tag_its)))
        else:
            it = sorted_merge(*tag_its, key=lambda x: -x[1])
        if self.probing=='all':
            for id, (data, probs, *stats) in it:
                yield tuple(data), stats
        else: #probing=='align'
            for id, (data, probs, *stats) in it:
                yield (id, tuple(data)), stats
    
    def _insert(self, log_probs0, log_probs1, index, number, alternate_tags):
        generator = self._iter_key(log_probs0, log_probs1, alternate_tags)
        for key, stats in  islice(generator, number):
            self.table.insert(key, index)
        return self
        
    def batch_insert(self, documents, indexes, number=None, alternate_tags=None):
        r"""
        Parameters
        ----------
        documents : torch.Tensor or np.ndarray (dtype: float, ndim: 2 or 3)
            The logits (i.e., before applying the sigmoid) of the Multi-Bernoulli. The first dimension
            corresponds to the batch, the last corresponds to the bits and if ndim==3, the second dimension
            is for multiple Multi-Bernoulli. 
        indexes : iterable of hashable
            A sequence of unique identifier for each corresponding document.
        number : None or int (optional)
            The number of outcomes to insert. If None, self.insert_number will be used. (default: None)
        alternate_tags : None or bool (optional)
            Overwrite the insert_alternate_tags. (default: None)
            
        Returns
        -------
        self : BernoulliMultiProbing
        
        Raises
        ------
        TypeError
            If documents.dtype is not a float.
        ValueError
            If len(documents) != len(indexes).
        """
        if len(documents) != len(indexes):
            msg = f'documents and indexes must have the same length, got {len(documents)} and {len(indexes)} respectively.'
            raise ValueError(msg)
        
        log_probs0, log_probs1 = self._parse_input(documents)
        number = self.insert_number if number is None else number
        alternate_tags = self.insert_alternate_tags if alternate_tags is None else alternate_tags
        for lp0, lp1, index in zip(log_probs0, log_probs1, indexes):
            self._insert(lp0, lp1, index, number=number, alternate_tags=alternate_tags)
        return self
    
    def _itersearch(self, itersearch, lp0, lp1, alternate_tags, halt_cost,
                    yield_cost, yield_empty, yield_duplicates):
        old = set()
        for key, (swap, comp, size) in self._iter_key(lp0, lp1, alternate_tags):
            new = self.table.search(key)
            itersearch.cost += self.cost(swap, comp, size)
            if not yield_duplicates:
                new = new - old
                old.update(new)
            if new or yield_empty:
                yield (new, itersearch.cost) if yield_cost else new
            if itersearch.cost >= halt_cost:
                break
                
    
    def _search(self, log_probs0, log_probs1, number, alternate_tags):
        generator = Itersearch(
                self._itersearch,
                lp0=log_probs0, lp1=log_probs1,
                alternate_tags=alternate_tags,
                halt_cost=np.inf, #no early stopping in search!
                yield_cost=False,
                yield_empty=True, #to also count the hash miss in islice
                yield_duplicates=False, #does not matter since we do the union
        )
        return set.union(*islice(generator, number))
    
    def batch_search(self, queries, number=None, alternate_tags=None):
        r"""
        Parameters
        ----------
        queries : torch.Tensor or numpy.ndarray(dtype: float, ndim: 2 or 3)
            The logits (i.e., before applying the sigmoid) of the Multi-Bernoulli. The first dimension
            corresponds to the batch, the last corresponds to the bits and if ndim==3, the second dimension
            is for multiple Multi-Bernoulli. 
        number : None or int (optional)
            The number of outcomes to search for in the hash table. If None,
            self.search_number will be used. (default: None)
        alternate_tags : None or bool (optional)
            Overwrite the search_alternate_tags. (default: None)
            
        Returns
        -------
        indexes : list of sets
            Each set contains the retrieved documents' identifier of the corresponding query.
        
        Raises
        ------
        TypeError
            If queries.dtype is not a float.
        """
        log_probs0, log_probs1 = self._parse_input(queries)
        number = self.search_number if number is None else number
        alternate_tags = self.insert_alternate_tags if alternate_tags is None else alternate_tags
        return [self._search(lp0, lp1, number, alternate_tags) for lp0, lp1 in zip(log_probs0, log_probs1)]
    
    def batch_itersearch(self, queries, alternate_tags=None, halt_cost=None,
                         yield_cost=False, yield_empty=False, yield_duplicates=False):
        r"""
        Parameters
        ----------
        queries : torch.Tensor or numpy.ndarray(dtype: float, ndim: 2 or 3)
            The logits (i.e., before applying the sigmoid) of the Multi-Bernoulli. The first dimension
            corresponds to the batch, the last corresponds to the bits and if ndim==3, the second dimension
            is for multiple Multi-Bernoulli. 
        alternate_tags : None or bool (optional)
            Overwrite the search_alternate_tags. (default: None)
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
            If queries.dtype is not a float.
        """
        log_probs0, log_probs1 = self._parse_input(queries)
        alternate_tags = self.insert_alternate_tags if alternate_tags is None else alternate_tags
        halt_cost = self.halt_cost if halt_cost is None else halt_cost
        return [
            Itersearch(
                self._itersearch,
                lp0=lp0, lp1=lp1,
                alternate_tags=alternate_tags,
                halt_cost=halt_cost,
                yield_cost=yield_cost,
                yield_empty=yield_empty,
                yield_duplicates=yield_duplicates,
            ) for lp0, lp1 in zip(log_probs0, log_probs1)
        ]
    
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
            'insert_number': self.insert_number,
            'search_number': self.search_number,
            'halt_cost': self.halt_cost,
            'table': self.table.get_state(),
        }
    
    def set_state(self, state):
        """
        Parameters
        -------
        state : dict
            The hash table to use
        """
        self.insert_number = state['insert_number']
        self.search_number = state['search_number']
        self.halt_cost = state['halt_cost']
        self.table.set_state(state['table'])
        return self
    
    def clear(self):
        """
        Clear the hash table
        """
        self.table.clear()
        return self