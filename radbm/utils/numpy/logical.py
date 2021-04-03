import numpy as np

def set_isrepeat(x):
    """
    Parameters
    ----------
    x : numpy.ndarray (shape: (n, k))

    Returns
    -------
    z : numpy.ndarray (shape: (n,), dtype: bool)
        z[i] is True iif x[i]'s elements are not unique
    
    Notes
    -----
    Works in O(n*k). A bit slower when many repeats.
    """
    return np.array([len(set(a))!=len(a) for a in x])

def vec_isrepeat(x):
    """
    Parameters
    ----------
    x : numpy.ndarray (shape: (n, k))

    Returns
    -------
    z : numpy.ndarray (shape: (n,), dtype: bool)
        z[i] is True iif x[i]'s elements are not unique
        
    Notes
    -----
    Vectorized implementation, works in O(n*k^2). Faster than set_isrepeat if k < 64
    since no hashing overhead and maximize numpy usage (vs python loop).
    """
    n, k = x.shape
    eye = np.eye(k, dtype=bool)[None]
    eq = x[:,None,:]==x[:,:,None]
    return np.logical_xor(eye, eq).any(axis=(1,2))

def isrepeat(x):
    """
    Parameters
    ----------
    x : numpy.ndarray (shape: (n, k))

    Returns
    -------
    z : numpy.ndarray (shape: (n,), dtype: bool)
        z[i] is True iif x[i]'s elements are not unique
        
    Notes
    -----
    Uses vec_isrepeat if k < 64 and set_isrepeat otherwise.
    """
    n, k = x.shape
    return vec_isrepeat(x) if k < 64 else set_isrepeat(x)

def set_issubset(x, y):
    """ 
    Parameters
    ----------
    x : numpy.ndarray (shape: (n, l))
    y : numpy.ndarray (shape: (n, k))

    Returns
    -------
    z : numpy.ndarray (shape: (n,), dtype: bool)
        z[i] is True iif x[i]'s elements are subset of y[i]'s elements.
        
    Notes
    -----
    Works in O(n*(l+k)).
    """
    return [set(a).issubset(b) for a, b in zip(x, y)]

def vec_issubset(x, y):
    """
    Parameters
    ----------
    x : numpy.ndarray (shape: (n, l))
    y : numpy.ndarray (shape: (n, k))

    Returns
    -------
    z : numpy.ndarray (shape: (n,), dtype: bool)
        z[i] is True iif x[i]'s elements are subset of y[i]'s elements.
        
    Notes
    -----
    Vectorized implementation, works in O(n*l*k). Faster than set_issubset for l < 128.
    """
    eq = x[:,:,None]==y[:,None,:]
    return eq.any(axis=2).all(axis=1)

def issubset(x, y):
    """
    Parameters
    ----------
    x : numpy.ndarray (shape: (n, l))
    y : numpy.ndarray (shape: (n, k))

    Returns
    -------
    z : numpy.ndarray (shape: (n,), dtype: bool)
        z[i] is True iif x[i]'s elements are subset of y[i]'s elements.
        
    Notes
    -----
    Uses vec_issubset if l < 128 and set_issubset otherwise.
    """
    n, l = x.shape
    return vec_issubset(x, y) if l < 128 else set_issubset(x, y)

def issubset_product_with_set(x, y):
    """
    Parameters
    ----------
    x : numpy.ndarray (shape: (n, k))
    y : numpy.ndarray (shape: (n, l))
        
    Returns
    -------
    issubset : list of list of int
        j (int) is in issubset[i] (list) iif set(x[i]).issubset(y[j]).
    """
    x = [set(v) for v in x]
    y = [set(w) for w in y]
    issubset = [sorted([n for n, w in enumerate(y) if v.issubset(w)]) for v in x]
    return issubset

def issubset_product_with_trie(x, y):
    """
    Parameters
    ----------
    x : numpy.ndarray (shape: (n, k))
    y : numpy.ndarray (shape: (n, l))
        
    Returns
    -------
    issubset : list of list of int
        j (int) is in issubset[i] (list) iif set(x[i]).issubset(y[j]).
        
    Notes
    -----
    Instead of doing len(queries)*len(documents) comparisons, this algorithm build a trie containing the queries.
    This is faster whenever the queries are sparse.
    """
    x = np.sort(x, axis=1)
    y = np.sort(y, axis=1)
    trie = dict()
    #filling the trie with x
    for n, v in enumerate(x):
        node = trie
        for e in v[:-1]:
            if e not in node:
                node[e] = dict()
            node = node[e]
        e = v[-1]
        if e not in node:
            node[e] = {n}
        node[e].add(n)
    
    issubset = [[] for v in x]
    #trie lookups
    for n, w in enumerate(y):
        nodes = [trie]
        subsets = set()
        while nodes:
            node = nodes.pop()
            if isinstance(node, set): #leaf
                subsets.update(node)
            else:
                nodes.extend([node[e] for e in w if e in node])
        for i in sorted(subsets):
            issubset[i].append(n)
    return issubset

def adjacency_list_to_matrix(adj_list, n=None):
    """
    Parameters
    ----------
    adj_list : list of list of int
        j in adj_list[i] indicates a link from i to j.
    n : int (optinal)
        The number of nodes (default: len(adj_list))
    
    Returns
    -------
    adj_matrix : numpy.ndarray (shape: (len(adj), n), dtype: bool)
        adj_matrix[i, j] is True iff there is a link from i to j.
        
    Notes
    -----
    The maximal value in adj_list needs to be smaller than n.
    """
    l = len(adj_list)
    n = l if n is None else n
    adj_matrix = np.zeros((l, n), dtype=bool)
    d0 = np.repeat(range(l), [len(r) for r in adj_list])
    d1 = [i for r in adj_list for i in r] #flatten relevants
    adj_matrix[(d0, d1)] = True
    return adj_matrix