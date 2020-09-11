def unique_list(it):
    """
    Create a list from an iterable with only unique element and where
    the order is preserved.
    
    Parameters
    ----------
    it : iterable
        Items should be hashable and comparable
        
    Returns : list
        All items in the list is unique and the order of the iterable
        is preserved.
    """
    unique = set()
    return [i for i in it if i not in unique and unique.add(i) is None]

def Ramp(x0, x1, y0, y1):
    """
    Parameters
    ----------
    x0 : float
        The input value where we start ramping
    x1 : float
        The input value where we stop ramping
    y0 : float
        The output value where the ramp starts
    y1 : float
        The output value where the ramp stops
    
    Returns
    -------
    ramp : function float -> float
        The ramping function
    """
    rate = (y1-y0) / (x1-x0)
    def ramp(x):
        if x is None: return y1
        y = (x-x0)*rate + y0
        a, b = x < x0, x1 <= x
        return a*y0 + b*y1 + (1-a)*(1-b)*y
    return ramp