import numpy as np
from radbm.utils.time import Chronometer
from radbm.utils.stats import superdupergeometric_expectations

def oracle_expectation_by_recalls(recalls, keys, probs, values):
    """
    An utility function meant for a very specific usage, this
    compute the (maximum likelihood) expectation estimator for
    each recalls for values of a precise format.
    
    Parameters
    ----------
    recalls : iterable of float in [0,1]
        The recalls for which we want the expectations.
        
    keys : ndarray
        1D with dtype=int containing all (unique) possible number
        of relevant documents.
        
    probs : ndarray
        1D with len(probs)=len(keys) and sum(probs)==1. probs[i] is
        the estimated probability that the keys[i] is the number of
        relevant document. 
        
    values : dict (k int -> v list)
        With k the number of relevant document and v[i] is the estimated
        expectation of the oracle for the number of revelant document=k
        and a wanted recall of i/k.
    
    Returns
    -------
        expectations : ndarray
            1D with dtype=float. expectations[i] is the expectation for an
            oracle stopping at a recall of at least recalls[i].
    """
    expectations = list()
    for recall in recalls:
        idxs = np.ceil(keys*recall).astype(np.int)
        e = sum(p*values[k][i] for i,k,p in zip(idxs,keys,probs))
        expectations.append(e)
    return np.array(expectations)

def oracle_random_calls_expectation_by_recall(recalls, keys, probs, N):
    """
    This function gives the expected number of calls (by query) to an oracle we
    would need to make to get a desired recall by trying documents at random.
    
    This is a baseline to any algorithm producing an order over the documents for
    the oracle to try. An algorithm producing an order on which the oracle would
    need (in expectation) more calls than what is returned by this function 
    [to achieve the same recall] as no value [for this particular recall].
    
    Parameters
    ----------
    recalls : numpy.ndarray
        The recalls for which we want the expected numbers
        
    keys : ndarray
        1D with dtype=int containing all (unique) possible number
        of relevant documents.
        
    probs : ndarray
        1D with len(probs)=len(keys) and sum(probs)==1. probs[i] is
        the estimated probability that the keys[i] is the number of
        relevant document. 
        
    N : int
        The total number of documents
    Returns
    -------
    expectations : numpy.ndarray
        The expected number of oracle calls for each recall given as input
        
    Notes
    -----
    How it works: For each query we have K relevant document out of N, the probability
    that the oracle find k relevant document in n samples is given by the superdupergeometric
    distribution SDGD. 
    This algorithm estimate (with maximum likelihood) the probability that their
    is K relevant documents for a query. Than for a specific recall r and for
    each k (with estimated pmf(K=k) > 0) we can find the mininum number of relevant
    documents to be found the get at least r of recall. 
    Finaly we estimate this expectation for a specific recall r with
    sum_{k s.t. pmf(K=k)>0} pmf(K=k) E[SDGD(N,k)[:, k_r]]
    """    
    expect_by_k = {k:superdupergeometric_expectations(N, k) for k in keys}
    expectations = oracle_expectation_by_recalls(recalls, keys, probs, expect_by_k)
    return expectations

def candidates_time_oracle_calls_count(candidates_iterator, relevant_set, recall_break=1):
    breaking_number = int(np.ceil(recall_break*len(relevant_set)))
    times = np.zeros((breaking_number+1,))
    calls = np.zeros((breaking_number+1,))
    calls_counter = 0
    founds_set = set()
    chronometer = Chronometer().start()
    for c in candidates_iterator:
        chronometer.stop()
        calls_counter += 1
        if c in relevant_set and not c in founds_set:
            founds_set.add(c)
            times[len(founds_set)] = chronometer.time()
            calls[len(founds_set)] = calls_counter
            if len(founds_set) == breaking_number:
                break #if candidates order is good measuring will be fast
        chronometer.start()
    if len(founds_set) != breaking_number:
        raise RuntimeError('Model did not found every document needed')
    return times, calls

def iter_candidates(model, query):
    for candidates in model.candidates(query):
        for candidate in candidates:
            yield candidate

def run_oracle_experiments(model, query_iterator, N, recalls=[1]):
    """
    This function computes the ToCR for each value in recalls.
    
    Parameters
    ----------
    model : IRModel
        model.candidates must be implemented and the model must
        already be indexing a set of documents.
        
    query_iterator : iterator
        (query, relevant_set) <- next(query_iterator) with query
        compatible with model.candidates and relevant_set containing
        the id of the relevant documents in the index of the model.
        
    N : int
        The number of documents in the index.
        
    recalls : iterable of float in [0, 1]
        The recalls for which we want to compute the ToCR.
        
    Returns
    -------
    expected_times : ndarray
        1D, shape=(len(recalls),), dtype=float. expected_times[i] correspond to
        the expected time it takes to iterate over enough candidates
        to get recall=recalls[i].
        
    expected_calls : ndarray
        1D, shape=(len(recalls),), dtype=float. expected_calls[i] correspond to
        the expected number of oracle calls it takes to iterate over enough
        candidates to get recall=recalls[i].
        
    expected_calls_baseline : ndarray
        1D, shape=(len(recalls),), dtype=float. expected_calls_baseline[i]
        correspond to the expected number of oracle calls it would take if
        we iterate randomly over all documents indexed to get recall=recalls[i].
        
     """
    times_dict = dict()
    calls_dict = dict()
    probs = dict()
    keys = list()
    recall_break = max(recalls)
    for query, relevant_set in query_iterator:
        candidates_iterator = iter_candidates(model, query)
        times, calls = candidates_time_oracle_calls_count(
            candidates_iterator, relevant_set, recall_break)
        k = len(relevant_set)
        if k in times_dict:
            times_dict[k].append(times)
            calls_dict[k].append(calls)
            probs[k] += 1
        else:
            keys.append(k)
            times_dict[k] = [times]
            calls_dict[k] = [calls]
            probs[k] = 1
            
    keys = np.array(keys)
    probs = np.array([probs[k] for k in keys])/sum(probs.values())
    times_dict = {k:np.array(v).mean(axis=0) for k,v in times_dict.items()}
    calls_dict = {k:np.array(v).mean(axis=0) for k,v in calls_dict.items()}
    
    expected_times = oracle_expectation_by_recalls(recalls, keys, probs, times_dict)
    expected_calls = oracle_expectation_by_recalls(recalls, keys, probs, calls_dict)
    expected_calls_baseline = oracle_random_calls_expectation_by_recall(recalls, keys, probs, N)
    
    return expected_times, expected_calls, expected_calls_baseline
        
def time_over_oracle_calls_reduction_metric(model, query_iterator, N, recalls=[1]):
    """
    This function computes the ToCR for each value in recalls.
    
    Parameters
    ----------
    model : IRModel
        model.candidates must be implemented and the model must
        already be indexing a set of documents.
        
    query_iterator : iterator
        (query, relevant_set) <- next(query_iterator) with query
        compatible with model.candidates and relevant_set containing
        the id of the relevant documents in the index of the model.
        
    N : int
        The number of documents in the index.
        
    recalls : iterable of float in [0, 1]
        The recalls for which we want to compute the ToCR.
        
    Returns
    -------
    ToCRs : ndarray
        1D, shape=(len(recalls),), dtype=float. ToCRs[i] correspond to
        the ToCR with recall=recalls[i].
     """
    exp = run_oracle_experiments(model, query_iterator, N, recalls=recalls)
    expect_times, expect_calls, expect_calls_baseline = exp
    expect_calls_reduction = expect_calls_baseline - expect_calls
    
    strictly_positive = 0 < expect_calls_reduction
    safe_div = np.where(strictly_positive, expect_calls_reduction, 1)
    ToCRs = np.where(strictly_positive, expect_times/safe_div, np.inf)
    return ToCRs