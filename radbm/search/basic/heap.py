#code derived from the standard heapq to accommodate our needs. 
#https://github.com/python/cpython/blob/main/Lib/heapq.py

from radbm.utils import identity
from radbm.search.base import BaseSDS

def rise(heap, pos, get_key=identity):
    swap_count, comp_count = 0, 0
    value = heap[pos]
    key = get_key(value)
    # Follow the path to the root, moving parents down until finding a place
    while pos > 0:
        i = (pos - 1) >> 1
        parent = heap[i]
        comp_count += 1
        if key < get_key(parent):
            swap_count += 1
            heap[pos] = parent
            pos = i
            continue
        break
    heap[pos] = value
    return swap_count, comp_count

def sink(heap, pos, get_key=identity):
    swap_count, comp_count = 0, 0
    endpos = len(heap)
    value = heap[pos]
    key = get_key(value)
    # Bubble up the smaller child until hitting a leaf.
    minpos = 2*pos + 1    # leftmost child position
    while minpos < endpos: #while there is a child
        # Set childpos to index of smaller child.
        minchild = heap[minpos]
        minchild_key = get_key(heap[minpos])
        rightpos = minpos + 1
        if rightpos < endpos:
            rightchild = heap[rightpos]
            rightchild_key = get_key(rightchild)
            comp_count += 1
            if rightchild_key < minchild_key:
                minpos = rightpos
                minchild = rightchild
                minchild_key = rightchild_key
        comp_count += 1
        if minchild_key < key:
            swap_count += 1
            heap[pos] = heap[minpos]
            pos = minpos
            minpos = 2*pos + 1
            continue
        break
    heap[pos] = value
    return swap_count, comp_count

class KeyValueHeap(BaseSDS, list):
    r"""
    A key-value heap. I.e, the objects stored in the heap are not (necessarily) those
    used for comparisons. Similar to the built-in sorted algorithm, A custom key function
    can be supplied to customize the sort order. The code of the class is an adaptation of the 
    standard python `heapq <https://github.com/python/cpython/blob/main/Lib/heapq.py>`__ algorithm
    that support the key keyword and returns the number of swaps and comparison of the insert and pop
    methods; useful for academic analysis.
    
    Parameters
    ----------
    *items : objects
        The initial to insert in the heap. The items themselve need not to be comparable. However,
        key(item1) < key(item2) must work.
    return_counts : bool (optional)
        Whether or not to return the number of swaps and comparisons in the pop method. Those quantities
        will always be returned in the batch_insert and insert methods. (default: False)
    key : function (optional)
        A function that takes an item an return a object (e.g. a float) that can be compared. (default: identity)
    """
    def __init__(self, *items, return_counts=False, key=identity):
        self.get_key = key
        self.return_counts = return_counts
        if items:
            self.batch_insert(items)
    
    def batch_insert(self, items, key=None):
        """
        A non-standard batch_insert methods. It does not take an indexes arguments.
        
        Parameters
        ----------
        items : iterables of objects
            The items to insert in the heap.
        key : function or None (optional)
            A function that takes an item an return a object (e.g. a float) that can be compared. If None,
            the key provided at initialization will be used. (default: None)
        Returns
        -------
        swap_count: int
            The total number of swaps performed in the heap while inserting every items.
        comp_count: int
            The total number of comparisons performed in the heap while inserting every items.
        """
        swap_count, comp_count = 0, 0
        for item in items:
            s, c = self.insert(item, key=key)
            swap_count += s
            comp_count += c
        return swap_count, comp_count
    
    def insert(self, *item, key=None):
        """
        A non-standard insert methods. It does not take an index arguments.
        
        Parameters
        ----------
        item : object
            The item to insert in the heap.
        key : function or None (optional)
            A function that takes an item an return a object (e.g. a float) that can be compared. If None,
            the key provided at initialization will be used. (default: None)
        Returns
        -------
        swap_count: int
            The total number of swaps performed in the heap while inserting the item.
        comp_count: int
            The total number of comparisons performed in the heap while inserting the item.
        """
        key = self.get_key if key is None else key
        self.append(*item)
        return rise(self, pos=len(self)-1, get_key=key)
            
    def batch_search(self, key=None):
        msg = "It does not make sense to perform a batch_search on a heap. The minimum always stay the same."
        raise NotImplementedError(msg)
    
    def search(self, key=None):
        return self[0]

    def pop(self, return_counts=None, key=identity):
        """
        Pop the smallest item off the heap, maintaining the heap invariant.
        
        Parameters
        ----------
        return_counts : boolean or None (optional)
            If True, the swap and comparison count will be given. If None,
            the return_counts provided at initialization will be used. (default: None)
        key : function or None (optional)
            A function that takes an item an return a object (e.g. a float) that can be compared. If None,
            the key provided at initialization will be used. (default: None)
        Returns
        -------
        minitem : object
            The smallest item in the heap
        swap_count: int (if return_counts is True)
            The total number of swaps performed in the heap while inserting the item.
        comp_count: int (if return_counts is True)
            The total number of comparisons performed in the heap while inserting the item.
        """
        key = self.get_key if key is None else key
        return_counts = self.return_counts if return_counts is None else return_counts
        lastelt = super().pop()    # raises appropriate IndexError if heap is empty
        if self:
            returnitem = self[0]
            self[0] = lastelt
            swap_count, comp_count = sink(self, pos=0, get_key=key)
            return (returnitem, swap_count, comp_count) if return_counts else returnitem
        return (lastelt, 0, 0) if return_counts else lastelt

    def sort(self, key=None):
        """
        Sort the item of the heap in linear time by popping them (O(n)) and reinserting in the heap.
        The new (sorted) order is still a heap and can still be used as such.
        
        Parameters
        ----------
        key : function or None (optional)
            A function that takes an item an return a object (e.g. a float) that can be compared. If None,
            the key provided at initialization will be used. (default: None)
        
        Returns
        -------
        self : KeyValueHeap
            With the items sorted.
        """
        self.overwrite([self.pop(key=key, return_counts=False) for _ in range(len(self))])
        return self
    
    def keys(self, key=None):
        """
        Iter over the keys in the heap (not in order) using the key method.
        
        Parameters
        ----------
        key : function or None (optional)
            A function that takes an item an return a object (e.g. a float) that can be compared. If None,
            the key provided at initialization will be used. (default: None)
        
        Yields
        ------
        k : comparable
            The result of key(item).
        """
        key = self.get_key if key is None else key
        for item in self:
            yield key(item)
    
    def get_state(self):
        return {
            'heap': self,
            'return_counts': self.return_counts,
            'get_key': self.get_key,
        }
    
    def overwrite(self, heap):
        """overwrite the heap, no check is performed to see if this is a heap."""
        self.clear()
        self.extend(heap)
    
    def set_state(self, state):
        self.overwrite(state['heap'])
        self.get_key = state['get_key']
        self.return_counts = state['return_counts']
        return self