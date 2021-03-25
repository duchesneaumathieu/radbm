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