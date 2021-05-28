import torch
import numpy as np
from itertools import islice
from radbm.search.base import BaseSDS
from radbm.search import DictionarySearch
from radbm.utils.torch import tuple_cast
from radbm.utils.generators import likeliest_multi_bernoulli_outcomes
from radbm.search.reduction.bernoulli import get_log_multi_bernoulli_probs

__torch_float = {torch.float16, torch.float32, torch.float64}
    
def _parse_multi_bernoulli_input(x, ndim):
    #If x is logits
    if isinstance(x, torch.Tensor):
        if x.dtype not in __torch_float:
            msg = ('If a torch.Tensor is given to BernoulliMultiProbing, it must be a float since it represents the logits '
                   '(pre-sigmoid) of the probabilities of each bit to be one.')
            raise TypeError(msg)
        if x.ndim != ndim:
            msg = (f'ndim is {x.ndim}, expected {ndim}.')
            raise ValueError(msg)
        log_probs0, log_probs1 = get_log_multi_bernoulli_probs(x) #the only reason why we don't want numpy
                                                                  #maybe we can generalized this later.
    #if x is tuple of log probabilities
    elif isinstance(x, (list, tuple)):
        if len(x) != 2:
            msg = f'If a tuple (or list) is given to BernoulliMultiProbing, the length must be two. Got {len(x)}.'
            raise ValueError(msg)
        log_probs0, log_probs1 = x
        if not isinstance(log_probs0, torch.Tensor) or not isinstance(log_probs1, torch.Tensor):
            msg = (f'When a tuple (or list), say x, is given to BernoulliMultiProb, x[0] and x[1] must be torch.Tensors. '
                   f'Got {type(log_probs0)} and {type(log_probs1)} respectively.')
            raise TypeError(msg)
        if log_probs0.ndim != ndim:
            msg = (f'ndim of the first Tensor is {log_probs0.ndim}, expected {ndim}.')
            raise ValueError(msg)
        if log_probs1.ndim != ndim:
            msg = (f'ndim of the second Tensor is {log_probs1.ndim}, expected {ndim}.')
            raise ValueError(msg)
        if log_probs0.dtype not in __torch_float or log_probs1.dtype not in __torch_float:
            msg = (f'When a tuple (or list) of torch.Tensor, say x, is given to BernoulliMultiProb, ' 
                   f'x[0] and x[1] must be float. Got {log_probs0.dtype} and {log_probs1.dtype} respectively.')
            raise TypeError(msg)
        if not torch.all(log_probs0<=0) or not torch.all(log_probs1<=0):
            msg = (f'When a tuple (or list) of torch.Tensor, say x, is given to BernoulliMultiProb, ' 
                   f'x[0] and x[1] must be negative since they must be log probabilities. '
                   f'However, positive value(s) where found.')
            raise TypeError(msg)
    else:
        msg = f'BernoulliMultiProbing takes torch.Tensor or a tuple of torch.Tensors. Got {str(type(x))}.'
        raise TypeError(msg)
        
    return log_probs0.detach().cpu().numpy(), log_probs1.detach().cpu().numpy()

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
    halt_cost : float (optional)
        The cost at which to halt when generating candidates in the itersearch and batch_itersearch methods. The value
        can be overwritten in those methods using the halt_cost keyword.
    cost : function (optional) (int swap, int comp, int size) -> float cost
        The cost function tells how the itertsearch show compute the engine cost. Swap, comp and size are all statistics
        from the heap.pop() and heap.push() methods used to compute the outcomes in decreasing order of probabilities.
        Swap is the number of heap swap, comp is the number of heap comparison, and size is the current heap size.
        (default: lambda swap,comp,size: swap+comp+1)
    """
    def __init__(self, insert_number=1, search_number=1, halt_cost=np.inf, cost=lambda swap,comp,size: swap+comp+1):
        self.insert_number = insert_number
        self.search_number = search_number
        self.halt_cost = halt_cost
        self.cost = cost
        self.table = DictionarySearch()
        
    def batch_insert(self, documents, indexes, number=None):
        r"""
        Parameters
        ----------
        documents : torch.Tensor (dtype: float, ndim: 2) or tuple of two Torch.tensor
            If torch.Tensor, the documents' logits, i.e., before applying the sigmoid. If tuple, the first tensor
            most be the log probabilities of each bit to be zero and the second tensor most be the log probabilities
            of each bit to be one.
        indexes : iterable of hashable
            A sequence of unique identifier for each corresponding document.
        number : None or int (optional)
            The number of outcomes to insert. If None, self.insert_number will be used. (default: None)
            
        Returns
        -------
        self : BernoulliMultiProbing
        
        Raises
        ------
        TypeError
            If documents is not a torch.Tensor nor a tuple (or list) of two torch.Tensor
        ValueError
            If the torch.Tensor(s) ndim is not 2.
        TypeError
            If documents is torch.Tensor and documents.dtype is not float.
        ValueError
            If documents is a tuple (or a list) and len(documents) != 2.
        TypeError
            If documents is a tuple (or a list) and documents[0] or documents[1] are not torch.Tensor or
            they don't correspond to log probabilities (i.e., negative float).
        ValueError
            If documents and indexes don't have the same length
        """
        log_probs0, log_probs1 = _parse_multi_bernoulli_input(documents, ndim=2)
        if len(log_probs0) != len(indexes):
            msg = 'documents and indexes must have the same length, got {len(log_probs0)} and {len(indexes)} respectively.'
            raise ValueError(msg)
        
        number = self.insert_number if number is None else number
        for lp0, lp1, index in zip(log_probs0, log_probs1, indexes):
            self.insert((lp0, lp1), index, number=number, _check=False)
        return self
    
    def insert(self, document, index, number=None, _check=True):
        r"""
        Parameters
        ----------
        document : torch.Tensor (dtype: float, ndim: 1) or tuple of two torch.Tensor
            If torch.Tensor, the document' logits, i.e., before applying the sigmoid. If tuple, the first tensor
            most be the log probabilities of each bit to be zero and the second tensor most be the log probabilities
            of each bit to be one.
        index : hashable
            A unique identifier for the document.
        number : None or int (optional)
            The number of outcomes to insert. If None, self.insert_number will be used. (default: None)
            
        Returns
        -------
        self : BernoulliMultiProbing
        
        Raises
        ------
        TypeError
            If document is not a torch.Tensor nor a tuple (or list) of two torch.Tensor
        ValueError
            If the torch.Tensor(s) ndim is not 1.
        TypeError
            If document is torch.Tensor and document.dtype is not float.
        ValueError
            If document is a tuple (or a list) and len(document) != 2.
        TypeError
            If document is a tuple (or a list) and document[0] or document[1] are not torch.Tensor or
            they don't correspond to log probabilities (i.e., negative float).
        """
        if _check:
            document = _parse_multi_bernoulli_input(document, ndim=1)
            number = self.insert_number if number is None else number
        generator = islice(likeliest_multi_bernoulli_outcomes(*document), number)
        for outcome in generator:
            self.table.insert(tuple(outcome), index)
        return self
    
    def batch_search(self, queries, number=None):
        r"""
        Parameters
        ----------
        queries : torch.Tensor (dtype: float, ndim: 2) or tuple of two torch.Tensor
            If torch.Tensor, the queries' logits, i.e., before applying the sigmoid. If tuple, the first tensor
            most be the log probabilities of each bit to be zero and the second tensor most be the log probabilities
            of each bit to be one.
        number : None or int (optional)
            The number of outcomes to search for in the hash table. If None,
            self.insert_number will be used. (default: None)
            
        Returns
        -------
        indexes : list if sets
            Each set contains the retrieved documents' identifier of the corresponding query.
        
        Raises
        ------
        TypeError
            If queries is not a torch.Tensor nor a tuple (or list) of two torch.Tensor
        ValueError
            If the torch.Tensor(s) ndim is not 2.
        TypeError
            If queries is torch.Tensor and queries.dtype is not float.
        ValueError
            If queries is a tuple (or a list) and len(queries) != 2.
        TypeError
            If queries is a tuple (or a list) and queries[0] or queries[1] are not torch.Tensor or
            they don't correspond to log probabilities (i.e., negative float).
        """
        log_probs0, log_probs1 = _parse_multi_bernoulli_input(queries, ndim=2)
        number = self.search_number if number is None else number
        return [self.search((lp0, lp1), number=number, _check=False) for lp0, lp1 in zip(log_probs0, log_probs1)]
    
    def search(self, query, number=None, _check=True):
        r"""
        Parameters
        ----------
        query : torch.Tensor (dtype: float, ndim: 1) or tuple of two torch.Tensor
            If torch.Tensor, the queries' logits, i.e., before applying the sigmoid. If tuple, the first tensor
            most be the log probabilities of each bit to be zero and the second tensor most be the log probabilities
            of each bit to be one.
        number : None or int (optional)
            The number of outcomes to search for in the hash table. If None,
            self.insert_number will be used. (default: None)
            
        Returns
        -------
        indexes : set
            The retrieved documents' identifier for this query.
        
        Raises
        ------
        TypeError
            If query is not a torch.Tensor nor a tuple (or list) of two torch.Tensor
        ValueError
            If the torch.Tensor(s) ndim is not 1.
        TypeError
            If query is torch.Tensor and query.dtype is not float.
        ValueError
            If query is a tuple (or a list) and len(query) != 2.
        TypeError
            If query is a tuple (or a list) and query[0] or query[1] are not torch.Tensor or
            they don't correspond to log probabilities (i.e., negative float).
        """
        generator = self.itersearch(
            query,
            halt_cost=np.inf, #no early stopping in search!
            yield_empty=True, #to also count the hash miss in islice
            _check=_check
        )
        if _check:
            number = self.search_number if number is None else number
        return set.union(*islice(generator, number))
    
    def itersearch(self, query, halt_cost=None, yield_cost=False, yield_empty=False, yield_duplicates=False, _check=True):
        r"""
        Parameters
        ----------
        query : torch.Tensor (dtype: float, ndim: 1) or tuple of two torch.Tensor
            If torch.Tensor, the queries' logits, i.e., before applying the sigmoid. If tuple, the first tensor
            most be the log probabilities of each bit to be zero and the second tensor most be the log probabilities
            of each bit to be one.
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
            If query is not a torch.Tensor nor a tuple (or list) of two torch.Tensor
        ValueError
            If the torch.Tensor(s) ndim is not 1.
        TypeError
            If query is torch.Tensor and query.dtype is not float.
        ValueError
            If query is a tuple (or a list) and len(query) != 2.
        TypeError
            If query is a tuple (or a list) and query[0] or query[1] are not torch.Tensor or
            they don't correspond to log probabilities (i.e., negative float).
        """
        if _check:
            query = _parse_multi_bernoulli_input(query, ndim=1)
            halt_cost = self.halt_cost if halt_cost is None else halt_cost
        old = set()
        total_cost = 0.
        for outcome, swap, comp, size in likeliest_multi_bernoulli_outcomes(*query, yield_stats=True):
            new = self.table.search(tuple(outcome))
            total_cost += self.cost(swap, comp, size)
            if not yield_duplicates:
                new = new - old
                old.update(new)
            if new or yield_empty:
                yield (new, total_cost) if yield_cost else new
            if total_cost >= halt_cost:
                break
    
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