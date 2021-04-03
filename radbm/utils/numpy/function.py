import numpy as np

def sum_log_range(a, b):
    return np.log(np.arange(a, b)).sum()

def log_comb(n, k):
    """
    Numerically stable version of numpy.log(scipy.special.comb(n, k))
    """
    k = max(k, n-k) #simple speedup.
    if k > n: return -np.inf
    return sum_log_range(k+1, n+1) - sum_log_range(1, n-k+1)

def softplus(x):
    """
    Numerically stable numpy.log(1 + numpy.exp(x))
    """
    return np.maximum(0, x) + np.log(1+np.exp(-np.abs(x)))

def softplusinv(x):
    """
    Numerically stable numpy.log(numpy.exp(x) - 1)
    
    Notes
    -----
    The domain of this function is the strictly positive reals.
    """
    return np.maximum(0, x) + np.log(1-np.exp(-x))

#Dihedral group of order 4 (for images)
def sr3(z):
    return z.swapaxes(-1,-2) #swap x and y

def sr2(z):
    return z[...,::-1,:] #reflection along the y (height) axis

def s(z):
    return z[...,:,::-1] #reflection along the x (width) axis

#we have Time(sr3) < Time(sr2) < Time(s) (it is not significant but useful to choose the formulae)
_transforms_map = {
    'r0': [], #i.e. doing nothing
    #'r1': [s, sr3], #sr3s = r(-3) = r1
    'r1': [sr3, sr2], #sr2sr3 = r(-2)r3 = r2r3 = r1
    'r2': [s, sr2], #sr2s = r(-2) = r2
    #'r2': [sr2, s], #ssr2 = r2
    #'r3': [sr3, s], #sr3sr2 = r(-3)r2 = r1r2 = r3
    'r3': [sr2, sr3], #sr3sr2 = r(-3)r2 = r1r2 = r3
    'sr0': [s],
    #'sr1': [s, sr3, s], #ssr3s = sr(-3) = sr1
    'sr1': [sr3, sr2, s], #ssr2sr3 = sr(-2)r3 = sr2r3 = sr1
    'sr2': [sr2],
    'sr3': [sr3],
}

def dihedral4(x, transform='r0'):
    """
    Parameters
    ----------
    x : numpy.ndarray (ndim: >=2)
        The last 2 dims are interpreted as the y (height) and x (width) axes respectively. The y axis goes from up to bottom 
        while the x axis goes from left to right (as in images).
    transform : str (optional)
        Should be 'r0', 'r1', 'r2', 'r3', 'sr0', 'sr1', 'sr2', 'sr3'.
        Where r is a 90 degree rotation and s is the vertical flip. The operations starts
        at the right e.g. 'sr2' means a rotation of 180 degree and then a vertical flip.
        (default: 'r0' the identity i.e. doing nothing)
    """
    if transform not in _transforms_map:
        msg = "transform should be in {{'r0', 'r1', 'r2', 'r3', 'sr0', 'sr1', 'sr2', 'sr3'}}, got transform = '{}'"
        raise ValueError(msg.format(transform))
    for f in _transforms_map[transform]:
        x = f(x)
    return x