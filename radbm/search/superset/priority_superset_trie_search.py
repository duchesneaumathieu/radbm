from radbm.search.base import BaseSDS, Itersearch
from radbm.search.basic import DictionarySearch, Trie
from .superset_trie_search import residual_suffix
            
def limitrophes_suffix(suffix, prefix):
    #find what prevents suffix to be a subset of a node in the sub-trie of prefix.
    limit = ()
    while True:
        if not suffix or not prefix:
            return limit
        elif suffix[0] < prefix[0]:
            limit += (suffix[0],)
            suffix = suffix[1:]
        elif suffix[0] == prefix[0]:
            suffix = suffix[1:]
            prefix = prefix[1:]
        elif suffix[0] > prefix[0]:
            #this cannot happen in PrioritySupersetTrieSearch._itersearch
            #since prefix is the branch and while never contain element not in suffix
            prefix = prefix[1:]

class PrioritySupersetTrieSearch(BaseSDS):
    r"""
    This implement a variant of the superset search algorithm described in `A New Method to Index and Query Sets 
    <https://www.ijcai.org/Proceedings/99-1/Papers/067.pdf>`__ where the documents and are indexed by sets but where 
    the queries provide a priority over each element in the set. When searching, the algorithm will assume the query
    contains every possible element to search in the trie. After this exploration is done, the search algorithm will
    remove the element with the least priority and resume the exploration of the trie. This repeat until every document
    is generated or up to a halt_cost.
    
    Notes
    -----
    The documents should be iterable of int in {0, 1, ..., n} where n is the number of possible elements.
    
    Parameters
    ----------
    halt_cost : float (optional)
        At what cost we stop searching. Each node considered cost 1 and looking inside the hash table cost 1. 
    search_type : str (optional)
        Should be either 'dfs' or 'bfs' for depth first search or breadth first search respectively.
        This only affects the itersearch method by changing the order the document will be outputed.
        This does not affects the search method since it wait for every relevant node to be explored.
        (default: 'dfs')
    """
    def __init__(self, halt_cost=float('inf'), search_type='dfs'):
        self.valid_search_type = ['dfs', 'bfs']
        self.halt_cost = halt_cost
        self.search_type = search_type
        self.popside = self.get_popside(self.search_type)
        self.table = DictionarySearch()
        self.trie = Trie()
        
    def get_popside(self, search_type):
        if search_type not in self.valid_search_type:
            raise ValueError(f'search_type must be in {self.valid_search_type}, got {search_type}.')
        return 0 if search_type == 'bfs' else -1
    
    def insert(self, document, index):
        r"""
        Parameters
        ----------
        document : iterable of hashable with order (e.g. int or str)
            The document's code
        indexes : hashable
            A unique identifier for the document.
            
        Returns
        -------
        self : PrioritySupersetTrieSearch
        """
        document = tuple(sorted(set(map(int, document))))
        self.table.insert(document, index)
        self.trie.insert(document)
        
    def _itersearch_output(self, itersearch, old, prefix, yield_duplicates, yield_empty, yield_cost):
        new = self.table.search(prefix)
        itersearch.cost += 1
        if not yield_duplicates:
            new = new - old
            old.update(new)
        if new or yield_empty:
            yield (new, itersearch.cost) if yield_cost else new

    def _itersearch(self, itersearch, query, halt_cost, yield_cost, 
                    yield_empty, yield_duplicates, popside, start_phase):
        n = len(query)
        full = tuple(range(n))
        explr_buffers = list([] for _ in range(n))
        yield_buffers = list([] for _ in range(n))
        nodes = [(query.argmax(), full, self.trie.root)]
        
        old = set()
        argsort = query.argsort().tolist()
        completed_phases = set(argsort[:start_phase])
        for phase in argsort[start_phase:]:
            nodes += explr_buffers[phase]
            for node in yield_buffers[phase]:
                yield from self._itersearch_output(
                    itersearch, old, node.prefix, yield_duplicates, yield_empty, yield_cost)
            completed_phases.add(phase)
            while nodes:
                yield_phase, suffix, node = nodes.pop(popside)
                prefix = node.prefix
                if yield_phase in completed_phases and (node.tag or yield_empty):
                    yield from self._itersearch_output(
                        itersearch, old, node.prefix, yield_duplicates, yield_empty, yield_cost)
                elif node.tag or yield_empty: yield_buffers[yield_phase].append(node)
                for k, v in node.items():
                    k_set = set(k)
                    k_complement = [i for i in suffix if i not in k_set]
                    yield_phase = max(k_complement, key=query.__getitem__, default=None)
                    limit = limitrophes_suffix(suffix, k)
                    explr_phase = max(limit, key=query.__getitem__, default=None)
                    new_suffix = residual_suffix(suffix, k, skip=True)
                    if explr_phase is None or explr_phase in completed_phases:
                        nodes.append((yield_phase, new_suffix, v))
                    else:
                        explr_buffers[explr_phase].append((yield_phase, new_suffix, v))
                    itersearch.cost += 1
                if itersearch.cost >= halt_cost:
                    return
    
    def itersearch(self, query, halt_cost=None, yield_cost=False, yield_empty=False,
                   yield_duplicates=False, search_type=None, start_phase=0):
        r"""
        Parameters
        ----------
        query : torch.Tensor or numpy.ndarray(dtype: float, ndim: 2)
            The priorities of each element's presence in the query's set. The search will start with every element present
            (except whan start_phase is provided and greater than zero) and elements will be remove one at a time
            starting with the element having the less (smaller) priority to further search in the trie.
        halt_cost : None or float (optional)
            The cost at which to halt. If None, self.halt_cost will be used. (default: None)
        yield_cost : bool (optional)
            Whether to give the total cost each time a set of candidates is yielded. (default: False)
        yield_empty : bool (optional)
            Whether to yield an empty set when we explore a possible node which could hold a superset the
            data was inserted. (default: False)
        yield_duplicates : bool (optional)
            Whether to yield documents' identifier each time they are found. Even if they were found before.
            (default: False)
        search_type : None, str (optional)
            It overwrite the search_type set at initialization. If None it self.search_type will be used. (default: 'dfs')
        start_phase : int (optional)
            How many element to remove from the query's set before exploring. (default: 0)
            
        Yields
        -------
        indexes : set (if yield_cost is False (default))
             A set containing the retrieved documents' identifier of the corresponding query.
        indexes, cost : set, float (if yield_cost is True)
            If yield_cost is True, tuples will be yieled where costs is the total cost of the search so far.
            
        Raises
        ------
        ValueError
            If search_type not 'dfs' nor 'bfs'.
        """
        halt_cost = self.halt_cost if halt_cost is None else halt_cost
        search_type = self.search_type if search_type is None else search_type
        popside = self.get_popside(search_type)
        return Itersearch(
                self._itersearch,
                query=query,
                halt_cost=halt_cost,
                yield_cost=yield_cost,
                yield_empty=yield_empty,
                yield_duplicates=yield_duplicates,
                popside=popside,
                start_phase=start_phase,
            )
    
    def search(self, query, halt_cost=None):
        r"""
        Parameters
        ----------
        query : torch.Tensor or numpy.ndarray(dtype: float, ndim: 2)
            The priorities of each element's presence in the query's set. The search will start with every element present
            (except whan start_phase is provided and greater than zero) and elements will be remove one at a time
            starting with the element having the less (smaller) priority to further search in the trie.
        halt_cost : None or float (optional)
            The cost at which to halt. If None, self.halt_cost will be used. (default: None)
            
        Returns
        -------
        indexes : set
            A set containing the retrieved documents' identifier of the corresponding query.
        """
        halt_cost = self.halt_cost if halt_cost is None else halt_cost
        return set().union(*self.itersearch(query, halt_cost=halt_cost))