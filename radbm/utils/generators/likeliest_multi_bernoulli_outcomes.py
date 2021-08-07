import numpy as np
from .smallest_subset_sums import smallest_subset_sums

def likeliest_multi_bernoulli_outcomes(log_probs0, log_probs1, yield_log_probs=False, yield_stats=False):
    """
    Generator that yields the outcomes of a Multi-Bernoulli in decreasing
    order of probability. This work by reducing to the problem of generating
    the subset of a set in increasing order of their sum.

    Notes
    -----
    Bits probability must be in ]0,1[ (i.e. they cannot be zero or one)


    Parameters
    ----------
    log_probs0 : numpy.ndarray (ndim: 1)
        Log probabilities for bits to be zero.
    log_probs1 : numpy.ndarray (ndim: 1)
        Log probabilities for bits to be one.
    yield_log_probs : bool (optional)
        Whether to provide the log probabilities of each outcome.
    yield_stats : bool (optional)
        Whether to provide statistic about the search. If True, the function will also yield
        the number of swap, the number of comparison and the size of the heap used to generate
        the outcomes. (default: False)


    Yields
    ------
    outcome : numpy.ndarray (dtype: bool)
        The Multi-Bernoulli outcomes in decreasing order of prababilities.
    log_prob: float (if yield_log_probs is True)
        The log probability of the yielded outcome.
    swap_count : int (if yield_stats is True)
        The number of swap done to maintain the heap while generating this outcome.
    comp_count : int (if yield_stats is True)
        The number of comparison done to maintain the heap while generating this outcome.
    heap_size : int (if yield_stats is True)
        The current size of the heap (before adding the children of the yielded outcome).

    Raises
    ------
    ValueError
        If the probabilities of each bit to be one or zero does not sum to one.

    """
    if log_probs0.ndim != 1 or log_probs1.ndim != 1:
        msg = f'log_probs0.ndim and log_probs1.ndim must be one. Got {log_probs0.ndim} and {log_probs1.ndim} respectively.'
        raise ValueError(msg)

    total_probs = np.exp(log_probs0) + np.exp(log_probs1)
    if not np.allclose(total_probs, 1):
        idx = np.where(~np.isclose(total_probs, 1))[0][0]
        msg = ('np.exp(log_probs0) + np.exp(log_probs1) must be filled with ones. '
               f'However, the sum is {total_probs[idx]:.4f}... at position {idx}.')
        raise ValueError(msg)

    #reduction to the smallest subset sum problem
    diff = log_probs1 - log_probs0
    most_probable_outcome = 0 < diff
    most_probable_outcome_log_prob = np.where(most_probable_outcome, log_probs1, log_probs0).sum()
    flipped_diff = np.where(most_probable_outcome, diff, -diff)
    for idx, ssum, sc, cc, l in smallest_subset_sums(flipped_diff, yield_sums=True, yield_stats=True):
        outcome = most_probable_outcome.copy()
        outcome[idx] ^= True
        support = (most_probable_outcome_log_prob - ssum,) if yield_log_probs else ()
        if yield_stats:
            support += (sc, cc, l)
        yield (outcome, *support) if support else outcome