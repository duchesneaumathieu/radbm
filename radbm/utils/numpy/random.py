import numpy as np
from scipy.special import comb
from itertools import combinations
from radbm.utils.numpy.logical import isrepeat, issubset

def enough_int(low, high, k):
    if high-low < k:
        msg = 'k cannot be bigger than high-low, got {} and {}.'
        raise ValueError(msg.format(k, high-low))

def unique_randint_with_permutation(low, high, n, k, rng=np.random):
    """
    Similar to numpy.random.randint but where each rows have unique elements.
    
    Parameters
    ----------
    low : int
        Lowest (signed) integer to be drawn from the distribution.
    high : int
        Largest (signed) integer to be drawn from the distribution.
    n : int
        The (unsigned) number of rows (vectors) with unique elements.
    k : int 
        The (unsigned) size of each rows (vectors).
    rng : numpy.random.generator.Generator (optional)
        The generator used for sampling. (default: np.random)

    Returns
    -------
    samples : np.ndarray (shape: (n, k), dtype: int)
        For all i<n, samples[i,j] == samples[i,k] iif j==k.
        
    Notes
    -----
    This implementation uses rng.permutation. The memory and time usage are in O(n*k + r) and O(r)
    respectively, with r = high-low.
    """
    enough_int(low, high, k)
    return np.array([(rng.permutation(high-low)+low)[:k] for i in range(n)])

def unique_randint_with_shuffle(low, high, n, k, rng=np.random):
    """
    Similar to numpy.random.randint but where each rows have unique elements.
    
    Parameters
    ----------
    low : int
        Lowest (signed) integer to be drawn from the distribution.
    high : int
        Largest (signed) integer to be drawn from the distribution.
    n : int
        The (unsigned) number of rows (vectors) with unique elements.
    k : int 
        The (unsigned) size of each rows (vectors).
    rng : numpy.random.generator.Generator (optional)
        The generator used for sampling. (default: np.random)

    Returns
    -------
    samples : np.ndarray (shape: (n, k), dtype: int)
        For all i<n, samples[i,j] == samples[i,k] iif j==k.
        
    Notes
    -----
    This implementation uses rng.shuffle. The memory and time usage are in O(n*k + r) and O(r*k)
    respectively, with r = high-low. Because of the inplace shuffling this implementation is 
    slightly faster than unique_randint_with_permutation however it takes more spaces.
    """
    enough_int(low, high, k)
    r = np.arange(low, high)
    rstack = np.stack(n*[r])
    for i in range(n):
        rng.shuffle(rstack[i])
    return rstack[:,:k]

def unique_randint_with_choice(low, high, n, k, rng=np.random):
    #slower than permutation since choice uses permutation, see
    #https://github.com/numpy/numpy/blob/c1ce397565398dcf22eda23f37d3c77ffe8f9b13/numpy/random/mtrand.pyx#L990
    #using permutation directly removes some overhead.
    enough_int(low, high, k)
    return np.array([rng.choice(high-low, size=k, replace=False)+low for i in range(n)])

def unique_randint_with_randint(low, high, n, k, rng=np.random):
    """
    Similar to numpy.random.randint but where each rows have unique elements.
    
    Parameters
    ----------
    low : int
        Lowest (signed) integer to be drawn from the distribution.
    high : int
        Largest (signed) integer to be drawn from the distribution.
    n : int
        The (unsigned) number of rows (vectors) with unique elements.
    k : int 
        The (unsigned) size of each rows (vectors).
    rng : numpy.random.generator.Generator (optional)
        The generator used for sampling. (default: np.random)

    Returns
    -------
    samples : np.ndarray (shape: (n, k), dtype: int)
        For all i<n, samples[i,j] == samples[i,k] iif j==k.
        
    Notes
    -----
    This implementation uses rng.randint with rejection sampling (i.e. reject when samples are not unique). This runs
    in O(n*k) memory and expected O(n*k) if k << high-low. This is way faster than shuffle/permutation if 
    k << high-low because the probability of accepting is close to one. This probability can be computed using
    p = (np.arange(r-k+1, r+1)/r).prod() with r = high-low.
    
    This implementation is recursive, consequently if p is to low (e.g. k is near high-low) then this will raise a
    RecursionError.
    """
    enough_int(low, high, k)
    samples = rng.randint(low, high, (n, k))
    not_valid = isrepeat(samples)
    if any(not_valid):
        samples[not_valid] = unique_randint_with_randint(low, high, sum(not_valid), k, rng=rng)
    return samples

class PartialPermutation(object):
    def __init__(self):
        self.table = dict()
    
    def __getitem__(self, i):
        return self.table[i] if i in self.table else i
    
    def switch(self, i, j):
        v = self[i]
        self.table[i] = self[j]
        self.table[j] = v

def fast_unique_randint(low, high, n, k, rng=np.random):
    """
    Similar to numpy.random.randint but where each rows have unique elements.
    
    Parameters
    ----------
    low : int
        Lowest (signed) integer to be drawn from the distribution.
    high : int
        Largest (signed) integer to be drawn from the distribution.
    n : int
        The (unsigned) number of rows (vectors) with unique elements.
    k : int 
        The (unsigned) size of each rows (vectors).
    rng : numpy.random.generator.Generator (optional)
        The generator used for sampling. (default: np.random)

    Returns
    -------
    samples : np.ndarray (shape: (n, k), dtype: int)
        For all i<n, samples[i,j] == samples[i,k] iif j==k.
        
    Notes
    -----
    The fastest unique_randint algorithm. Runs in O(n*k) memory and O(n*k) times. However this python
    implementation (without any vectorization) is quite slow.
    """
    enough_int(low, high, k)
    samples = np.zeros((n, k), dtype=int)
    for i in range(n):
        p = PartialPermutation()
        for j in range(k):
            r = rng.randint(j, high-low)
            samples[i, j] = p[r]+low
            p.switch(j, r)
    return samples

def unique_randint(low, high, n, k, rng=np.random):
    """
    Similar to numpy.random.randint but where each rows have unique elements.
    
    Parameters
    ----------
    low : int
        Lowest (signed) integer to be drawn from the distribution.
    high : int
        Largest (signed) integer to be drawn from the distribution.
    n : int
        The (unsigned) number of rows (vectors) with unique elements.
    k : int 
        The (unsigned) size of each rows (vectors).
    rng : numpy.random.generator.Generator (optional)
        The generator used for sampling. (default: np.random)

    Returns
    -------
    samples : np.ndarray (shape: (n, k), dtype: int)
        For all i<n, samples[i,j] == samples[i,k] iif j==k.
        
    Notes
    -----
    When the acceptance probability is fine, unique_randint_with_randint is used.
    Otherwise, if high-low < 500K then unique_randint_with_permutation is used. Otherwise, 
    fast_unique_randint is used.
    """
    enough_int(low, high, k)
    r = high-low
    if (r/np.arange(r-k+1, r+1)).prod() < 5:
        return unique_randint_with_randint(low, high, n, k, rng=rng)
    elif r < 500_000: #random rule
        return unique_randint_with_permutation(low, high, n, k, rng=rng)
    else:
        return fast_unique_randint(low, high, n, k, rng=rng)
    
def no_subset_unique_randint(low, high, n, l, x, rng=np.random):
    """
    Similar to numpy.random.randint but where each rows have unique elements. Also,
    not all elements generated will be in x.
    
    Parameters
    ----------
    low : int
        Lowest (signed) integer to be drawn from the distribution.
    high : int
        Largest (signed) integer to be drawn from the distribution.
    n : int
        The (unsigned) number of rows (vectors) with unique elements.
    l : int 
        The (unsigned) size of each rows (vectors).
    x : numpy.ndarray (shape: (n, k))
        The no subset conditions. See the returned value for more information.
    rng : numpy.random.generator.Generator (optional)
        The generator used for sampling. (default: np.random)

    Returns
    -------
    samples : np.ndarray (shape: (n, k), dtype: int)
        For all i<n, samples[i,j] == samples[i,k] iif j==k. Furthermore, 
        there exists j s.t. samples[i, j] is not present in x[i]. 
        
    Notes
    -----
    This implementation is recursive, consequently if the probability of accepting is to low
    then this will raise a RecursionError. There should be at least one integer inside np.arange(low, high)
    that is not in x[i] for all i. 
    """
    samples = unique_randint(low, high, n, l, rng=rng)
    not_valid = issubset(samples, x)
    if any(not_valid):
        samples[not_valid] = no_subset_unique_randint(low, high, sum(not_valid), l, x[not_valid], rng=rng)
    return samples

def _uniform_n_choose_k_input_validation(n, k, t):
    if t > comb(n, k):
        msg = 'Cannot sample more than comb(n, k) = {} unique samples. However, got t = {}.'
        raise ValueError(msg.format(comb(n, k), t))

def uniform_n_choose_k_by_enumeration(n, k, t, rng=np.random):
    """
    Parameters
    ----------
    n : int
        The number of elements to choose from.
    k : int
        The number of elements to choose without replacement.
    t : int
        The number of distinct vectors of k elements.
    rng : numpy.random.generator.Generator
        The random number generator used to sample the k-subsets. (default: numpy.random)
    
    Returns
    -------
    samples : numpy.ndarray (shape: (t, k))
        t unique vectors of k unique integers from 0,...,n-1
        
    Raises
    ------
    ValueError
        When t > comb(n, k).
        
    Notes
    -----
    This algorithms works in O(comb(n, k)) as it enumerates all possible combination before sampling
    t from it without replacement.
    """
    _uniform_n_choose_k_input_validation(n, k, t)
    c = np.array(list(combinations(range(n), k)))
    samples = c[rng.choice(len(c), t, replace=False)]
    return samples

def uniform_n_choose_k_by_rejection(n, k, t, rng=np.random):
    """
    Parameters
    ----------
    n : int
        The number of elements to choose from.
    k : int
        The number of elements to choose without replacement.
    t : int
        The number of distinct vectors of k elements.
    rng : numpy.random.generator.Generator
        The random number generator used to sample the k-subsets. (default: numpy.random)
    
    Returns
    -------
    samples : numpy.ndarray (shape: (t, k))
        t unique vectors of k unique integers from 0,...,n-1
    
    Raises
    ------
    ValueError
        When t > comb(n, k).
        
    Notes
    -----
    This algorithms is faster than uniform_n_choose_k_by_enumeration whenever t << comb(n, k).
    Otherwise, uniform_n_choose_k_by_enumeration is preferable.
    """
    _uniform_n_choose_k_input_validation(n, k, t)
    h = set()
    samples = np.zeros((t, k), dtype=int)
    duplicates = np.arange(t)
    while True:
        samples[duplicates] = np.sort(unique_randint(0, n, len(duplicates), k, rng=rng), axis=1)
        duplicates = [n for n in duplicates if tuple(samples[n]) in h or h.add(tuple(samples[n]))]
        if not duplicates: break
    return samples