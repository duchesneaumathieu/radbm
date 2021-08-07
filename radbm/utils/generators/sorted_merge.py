from radbm.search import KeyValueHeap

def sorted_merge(*gens, key=lambda x: x):
    """
    Given n generators that yield in increasing order, this method
    build a new generator that yield all the generators in increasing order.
    
    Parameters
    ----------
    *gens : sequence of generators
        All generator must yield in increasing order.
    key : function (optional)
        Used to customize the sort order. Same as the built-in sorted. (default: identity)
        
    Yields
    ------
    k : int
        The index of the generator that yielded this value
    v : object
        The current minimal value of all generators.
    """
    gens = list(map(iter, gens)) #make sure everything is iterable
    heap = KeyValueHeap(key=lambda x: key(x[1]))
    for k, g in enumerate(gens):
        try: heap.insert((k, next(g)))
        except StopIteration: pass
    
    while heap:
        (k, v) = heap.pop()
        #yield from the same generator as much as possible
        #this avoid overusing the heap
        while True:
            yield (k, v)
            try: v = next(gens[k])
            except StopIteration: break #we exhausted this generator, nothing to do
            if heap and key(v) > key(heap[0][1]): #another generator yielded a smaller value
                heap.insert((k, v)) #save this generator-value pair for later
                break