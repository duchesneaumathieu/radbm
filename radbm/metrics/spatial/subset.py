import torch

def subset_distance(x, y, *args, **kwargs):
    """
    Compute the number of position True in x which are False in y. e.g.
    x = torch.tensor([True, True, False, False, True]) and y = torch.tensor([True, False, True, False, False])
    we have for i in [1, 4], x[i] is True and y[i] is False thus subset_distance(x, y) = len([1, 4]) = 2.
    
    Parameters
    ----------
    x : torch.Tensor or numpy.ndarray (dtype=torch.bool)
        Broacastable with y.
    y : torch.tensor or numpy.ndarray (dtype=torch.bool)
        Broacastable with x.
    *args
        Passed to sum.
    **kwargs
        Passed to sum. E.g we can use subset_distance(x, y, dim=-1) for torch.Tensor or
        subset_distance(x, y, axis=-1) for numpy.ndarray to compute the subset_distance on the last dim. 
    
    Returns
    -------
    out : torch.Tensor or numpy.ndarray (dtype=int)
        The shape of out is the braocasted shape between x and y.
    """
    return (x & ~y).sum(*args, **kwargs)