import numpy as np
from radbm.utils.time import Chronometer

class MonitorTime(object):
    def __init__(self, gen, eta=1):
        self.gen = gen
        self.eta = eta
        self.chrono = Chronometer()
    
    def __iter__(self):
        return self
    
    def __next__(self):
        self.chrono.start()
        try:
            item = next(self.gen)
        finally:
            self.chrono.stop()
        return item
    
    def get_value(self):
        return self.eta*self.chrono.time()
    
class MonitorNext(object):
    def __init__(self, gen, eta=1):
        self.gen = gen
        self.eta = eta
        self.count = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        item = next(self.gen)
        self.count += 1 #does not count when next exit
        return item
    
    def get_value(self):
        return self.eta*self.count

def MatchingOracleCost(N, K, k):
    """
    Returns
    ------
    cost : float
        The expected number of call it take to find k documents out of K in
        a database containing N documents (without replacement)
        
    Notes
    -----
    This function does not check if the inputs are valid i.e. N >= K >= k
    """
    return k*(N+1)/(K+1)

def _sswr_sanitize_delta_candidates(candidates, delta_candidates, on_duplicate_candidates):
    """
    Helper function for SSWR
    """
    delta_candidates = set(delta_candidates) #just in case
    intersection = delta_candidates.intersection(candidates)
    if intersection and on_duplicate_candidates=='raise':
        msg = 'duplicate candidates found {}'
        raise RuntimeError(msg.format(intersection))
    return delta_candidates - intersection
            

def SSWR(relevant, monitor, N, recall=1, allow_halt=False, on_duplicate_candidates='raise'):
    """
    The Sequential Search Work Ratio (SSWR) metric used for quick retrieval task
    this function use monitor has the delta generator and monitor.get_value() for the 
    the cost of generating all the previous delta candidates. On might consider using
    ChronoSSWR or CounterSSWR, which implement a precise monitor, instead.
    
    Parameters
    ----------
    relevant : set of index
        Corresponding to the set of elemement that need to be retrieved
    monitor : object
        Iterator of set of index and implementing get_value() -> float
    N : int
        The number of documents in the database
    recall : float in [0,1] (optional)
        The minimal percentage of relevant document that should be generated
        (default 1)
    allow_halt : bool (optional)
        Allow the generator not to necessarily all needed indexes. This is similar, but
        not equal, to the case where the generator produce every other indexes at one
        and stop. (default False)
    on_duplicate_candidates : str (optional)
        Should be 'raise' or 'ignore'. Set what should be done if the same index
        is generated twice. If 'raise', a RunetimeError will be raised. Otherwise,
        if 'ignore', the diplicated candidate(s) will be removed. (default 'raise') 
    
    Returns
    -------
        work_ratio : float (or (float, bool) if allow_halt)
            By default, the SSWR. If allow_halt is True it returns a tuple with the
            SSWR and a boolean indicating if the generator halted abruptly.
            
    Raises
    ------
    ValueError
        If on_duplicate_candidates not in {'raises', 'ignore'}.
    RuntimeError
        If on_duplicate_candidates=='raise' and an index is generated twice.
    LookupError
        If allow_halt is False and delta_generator stops without
        generating enough relevant documents.
            
    Notes
    -----
    the aforementioned index is something hashable (i.e. hash(index) exists) that
    can be used to identify uniquely each document in the database. It might be the
    document itself, but that would be memory heavy. The most common case is using
    integer but in some cases it might be practical to have more information, maybe
    packed in a tuple for example.
    """
    valid = {'raise', 'ignore'}
    if on_duplicate_candidates not in valid:
        msg = 'on_duplicate_candidate must be in {}, got {}'
        raise ValueError(msg.format(valid, on_duplicate_candidates))
    count = 0
    candidates = set()
    K = len(relevant)
    k = int(np.ceil(recall*K))
    n_relevant = 0
    n_exhaustive_calls = 0
    for delta_candidates in monitor:
        delta_candidates = _sswr_sanitize_delta_candidates(
            candidates, delta_candidates, on_duplicate_candidates)
        n_delta_relevant = len(delta_candidates.intersection(relevant))
        n_relevant += n_delta_relevant
        if k <= n_relevant:
            break
        n_exhaustive_calls += len(delta_candidates)
        candidates.update(delta_candidates)
    halt = k > n_relevant #exit loop without break
    if halt and not allow_halt:
        msg = ('with allow_halt=False, the delta_generator exited without '
               'producing enough relevant documents. Needed {}, got {}')
        raise LookupError(msg.format(k, n_relevant))
    if halt:
        #random oracle search on the rest
        delta_N = N - n_exhaustive_calls
        delta_K = K - n_relevant
        delta_k = k - n_relevant
    else:
        #random oracle search on the last delta_candidates
        delta_N = len(delta_candidates)
        delta_K = n_delta_relevant
        delta_k = k - (n_relevant - n_delta_relevant)
    nume_work = MatchingOracleCost(delta_N, delta_K, delta_k)
    nume_work += n_exhaustive_calls + monitor.get_value()
    deno_work = MatchingOracleCost(N, K, k)
    work_ratio = nume_work / deno_work
    return (work_ratio, halt) if allow_halt else work_ratio

def ChronoSSWR(relevant, delta_generator, N, eta=1, recall=1, allow_halt=False, on_duplicate_candidates='raise'):
    """
    The Chronometer Sequential Search Work Ratio (SSWR) metric used for quick retrieval task.
    It uses the generating time (in seconds) has a mesure of the work done by the delta_generator.
    
    Parameters
    ----------
    relevant : set of index
        Corresponding to the set of elemement that need to be retrieved
    delta_generator : generator of set of index
        This should generate set of candidates (index)
    N : int
        The number of documents in the database
    eta : positive float (optional)
        The proportion of importance between a generating time (seconds) and one oracle call.
        e.g. eta=2 implies that 1 generator second is equivalent to 2 oracle call. (default 1)
    recall : float in [0,1] (optional)
        The minimal percentage of relevant document that should be generated
        (default 1)
    allow_halt : bool (optional)
        Allow the generator not to necessarily all needed indexes. This is similar, but
        not equal, to the case where the generator produce every other indexes at one
        and stop. (default False)
    on_duplicate_candidates : str (optional)
        Should be 'raise' or 'ignore'. Set what should be done if the same index
        is generated twice. If 'raise', a RunetimeError will be raised. Otherwise,
        if 'ignore', the diplicated candidate(s) will be removed. (default 'raise')
    
    Returns
    -------
        work_ratio : float (or (float, bool) if allow_halt)
            By default, the SSWR. If allow_halt is True it returns a tuple with the
            SSWR and a boolean indicating if the generator halted abruptly.
            
    Raises
    ------
    ValueError
        If on_duplicate_candidates not in {'raises', 'ignore'}.
    RuntimeError
        If on_duplicate_candidates=='raise' and an index is generated twice
    LookupError
        If delta_generator stops without generating enough relevant documents 
    """
    monitor = MonitorTime(delta_generator, eta=eta)
    return SSWR(relevant, monitor, N, recall=recall, allow_halt=allow_halt, on_duplicate_candidates=on_duplicate_candidates)

def CounterSSWR(relevant, delta_generator, N, eta=1, recall=1, allow_halt=False, on_duplicate_candidates='raise'):
    """
    The Counter Sequential Search Work Ratio (SSWR) metric used for quick retrieval task.
    It uses the number of call to the generator has a mesure of the work done by the delta_generator.
    
    Parameters
    ----------
    relevant : set of index
        Corresponding to the set of elemement that need to be retrieved
    delta_generator : generator of set of index
        This should generate set of candidates (index)
    N : int
        The number of documents in the database
    eta : positive float (optional)
        The proportion of importance between one generator call and one oracle call.
        e.g. eta=2 implies that 1 generator call is equivalent to 2 oracle call. (default 1)
    recall : float in [0,1] (optional)
        The minimal percentage of relevant document that should be generated
        (default 1)
    allow_halt : bool (optional)
        Allow the generator not to necessarily all needed indexes. This is similar, but
        not equal, to the case where the generator produce every other indexes at one
        and stop. (default False)
    on_duplicate_candidates : str (optional)
        Should be 'raise' or 'ignore'. Set what should be done if the same index
        is generated twice. If 'raise', a RunetimeError will be raised. Otherwise,
        if 'ignore', the diplicated candidate(s) will be removed. (default 'raise')
    
    Returns
    -------
        work_ratio : float (or (float, bool) if allow_halt)
            By default, the SSWR. If allow_halt is True it returns a tuple with the
            SSWR and a boolean indicating if the generator halted abruptly.
            
    Raises
    ------
    ValueError
        If on_duplicate_candidates not in {'raises', 'ignore'}.
    RuntimeError
        If on_duplicate_candidates=='raise' and an index is generated twice
    LookupError
        If delta_generator stops without generating enough relevant documents 
    """
    monitor = MonitorNext(delta_generator, eta=eta)
    return SSWR(relevant, monitor, N, recall=recall, allow_halt=allow_halt, on_duplicate_candidates=on_duplicate_candidates)