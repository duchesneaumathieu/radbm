import torch

def superset_cost(x, y, dim=-1, keepdim=False):
    """
    Compute the number of position True in x which are False in y. e.g.
    x = torch.tensor([True, True, False, False, True]) and y = torch.tensor([True, False, True, False, False])
    we have for i in [1, 4], x[i] is True and y[i] is False thus superset_cost(x, y) = len([1, 4]) = 2.
    
    Parameters
    ----------
    x : torch.Tensor or numpy.ndarray (dtype=torch.bool)
        Broacastable with y.
    y : torch.tensor or numpy.ndarray (dtype=torch.bool)
        Broacastable with x.
    dim : int (optional)
        The dimension along which to compute the hamming distance. (default: -1)
    keepdim : bool (optional)
        Whether to keep the reduced dimension. (default: False)
    
    Returns
    -------
    out : torch.Tensor or numpy.ndarray (dtype=int)
        The shape of out is the broadcasted shape between x and y.
    """
    return (x & ~y).sum(dim=dim, keepdim=keepdim)