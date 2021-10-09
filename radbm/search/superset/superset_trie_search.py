from radbm.search.base import BaseSDS, Itersearch
from radbm.search.basic import DictionarySearch, Trie

def residual_suffix(suffix, prefix):
    while True:
        if not suffix or not prefix:
            return suffix
        elif suffix[0] < prefix[0]:
            return None
        elif suffix[0] == prefix[0]:
            suffix = suffix[1:]
            prefix = prefix[1:]
        elif suffix[0] > prefix[0]:
            prefix = prefix[1:]

class SupersetTrieSearch(BaseSDS):
    r"""
    This implement the superset search algorithm described in `A New Method to Index and Query Sets 
    <https://www.ijcai.org/Proceedings/99-1/Papers/067.pdf>`__ where the documents and queries
    are indexed by sets.
    
    Parameters
    ----------
    halt_cost : float (optional)
        The number of outcomes to insert. Each document's outcomes will map to the document's index
        in the hash table. Can be overwritten in the batch_insert and insert methods using the number
        keyword. (default: inf)
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
        self : SupersetTrieSearch
        """
        document = tuple(sorted(set(map(int, document))))
        self.table.insert(document, index)
        self.trie.insert(document)
        
    def _itersearch(self, itersearch, query, halt_cost, yield_cost, 
                    yield_empty, yield_duplicates, popside):
        query = tuple(sorted(set(map(int, query))))
        old = set()
        nodes = [(query, self.trie.root)]
        while nodes:
            suffix, node = nodes.pop(popside)
            prefix = node.prefix
            if not suffix and (node.tag or yield_empty):
                new = self.table.search(prefix)
                itersearch.cost += 1
                if not yield_duplicates:
                    new = new - old
                    old.update(new)
                if new or yield_empty:
                    yield (new, itersearch.cost) if yield_cost else new
            for k, v in node.items():
                itersearch.cost += 1
                new_suffix = residual_suffix(suffix, k)
                if new_suffix is not None:
                    nodes.append((new_suffix, v))
            if itersearch.cost >= halt_cost:
                return
    
    def itersearch(self, query, halt_cost=None, yield_cost=False, yield_empty=False, yield_duplicates=False, search_type=None):
        r"""
        Parameters
        ----------
        query : iterable of hashable with order (e.g. int or str)
            The query's code to search for superset.
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
                popside=popside
            )
    
    def search(self, query):
        r"""
        Parameters
        ----------
        query : iterable of hashable with order (e.g. int or str)
            The query's code to search for superset.
            
        Returns
        -------
        indexes : set
            A set containing the retrieved documents' identifier of the corresponding query.
        """
        return set().union(*self.itersearch(query, halt_cost=float('inf')))