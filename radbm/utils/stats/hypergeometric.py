import numpy as np

def hypergeometric(N, K):
    """
    This function compute the pmf of the Hypergeometric(N, K, n)
    for each possible value of n (i.e. n in {0,1,2,...,N})
    
    In the context where an urn with N marbles contains K white marbles,
    this function outputs an array P of shape (N+1,K+1) where
    P[i, j] correspond to the probabillity that with i samples without 
    replacement we select j white marbles.
    
    Parameters
    ----------
    N : int
        The number of marbles in the urn
    K : int
        The number of white marbles in the urn
        
    Returns
    -------
    P : numpy.ndarray
        P[i, j] is the probability that with i samples without replacement
        we select j white marbles. P.shape is (N+1, K+1)
        
    Notes
    -----
    This is equivalent to 
    np.array([scipy.stats.hypergeom(N, K, n).pmf(range(0,K+1)) for n in range(N+1)])
    but faster, if only a row is needed (e.g. P[i]) than using scipy is faster.
    
    This algorithm uses the following recursive formula
    P[i, j] = P[i-1, j]*(1-Q[i-1, j]) + P[i-1, j-1]*Q[i-1, j-1]
    with Q[i, j] = (K-j)/(N-i) the probability of sampling a white marble
    given that i marbles where sampled from which j were white
    """
    P = np.zeros((N+1, K+1))
    P[0,0] = 1 #P[0, j] = 1 if j==0 else 0
    P[N,K] = 1 #P[N, j] = 1 if j==K else 0
    
    #To compute Q We don't need to bother with
    #value > 1 since they will be multiplied by 0
    Q = (K-np.arange(K+1))[None]/(N-np.arange(N))[:,None]
    
    for n in range(1,N):
        P[n] = P[n-1]*(1-Q[n-1])
        P[n,1:] += P[n-1,:-1]*Q[n-1,:-1]
        
    return P
    
def superdupergeometric(N, K):
    """
    This is the scenario where we have an urn with N marbles
    and K of them are white. We sample from the urn without replacement
    until we obtain k white marbles. The function gives the probability
    that we need n samples to obtain k white marbles.
    
    Parameters
    ----------
    N : int
        The number of marbles in the urn
    K : int
        The number of white marbles in the urn
        
    Returns
    -------
    SP : numpy.ndarray
        SP[i, j] is the probability that it requires i samples without replacement
        to select j white marbles. SP.shape is (N+1, K+1)
        
    Notes
    -----
    Probabibly related to the negative hypergeometric distribution
    
    This uses the hypergeometric (hence the name) 
    SP[i, j] = P[i-1, j-1]*Q[i-1, j-1]
    where P = hypergeometric(N, K) and with 
    Q[i, j] = (K-j)/(N-i) the probability of sampling a white marble
    given that i marbles where sampled from which j were white
    """
    
    P = hypergeometric(N, K)
    SP = np.zeros((N+1, K+1))
    SP[0, 0] = 1 #we only need 0 sample to get 0 white marbles
    
    #To compute Q We don't need to bother with
    #value > 1 since they will be multiplied by 0
    Q = (K-np.arange(K+1))[None]/(N-np.arange(N))[:,None]
    
    for n in range(1,N+1):
        SP[n,1:] += P[n-1,:-1]*Q[n-1,:-1]
        
    return SP

def superdupergeometric_expectations(N, K):
    """
    This is the scenario where we have an urn with N marbles
    and K of them are white. We sample from the urn without replacement 
    until we obtain k white marbles. The function gives the  expected
    number of samples n requires to get k white marbles (for each k).
    
    Parameters
    ----------
    N : int
        The number of marbles in the urn
    K : int
        The number of white marbles in the urn
        
    Returns
    -------
    ESP : numpy.ndarray
        ESP[k] is the expected number of samples without replacement
        requires to get k white marbles. ESP.shape is (K+1,)
        
    Notes
    -----
    This is equivalent (but way faster) to (SP*np.arange(N+1)[:,None]).sum(axis=0)
    where SP = superdupergeometric(N, K)
    """
    ESP = (N+1)/(K+1)*np.arange(K+1)
    return ESP
