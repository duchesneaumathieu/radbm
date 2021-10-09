import torch, itertools
import numpy as np
logsigmoid = torch.nn.LogSigmoid()

from radbm.utils.torch import torch_logsumexp, torch_logsubexp

def _membership_log_terms(log_x0, log_x1, log_y0, log_y1, l, terms):
    log_total = None #total should start at 0 so log_total should be -inf, hence the None.
    for t in terms:
        if t == 0: continue #avoid empty combination
        for combination in itertools.combinations(range(l), t):
            log_p0 = log_x0 + log_y0[..., combination, :].sum(dim=-2)
            log_p1 = log_x1 + log_y1[..., combination, :].sum(dim=-2)
            log_p = torch_logsumexp(log_p0, log_p1).sum(dim=-1)
            log_total = log_p if log_total is None else torch_logsumexp(log_total, log_p)
    return log_total

class MultiBernoulliMembershipMatch(object):
    r"""
    Functional to compute the probability that a Multi-Bernoulli
    is a member of a set of mutually independent Multi-Bernoulli.

    .. math::
        A_1 - A_2 + A_3 - A_4 + \dots \pm A_l

    where 

    .. math::
        A_t = \sum_{i_1=t}^l \sum_{i_2=t-1}^{i_1} \sum_{i_3=t-2}^{i_2} \dots \sum_{i_t=1}^{i_{t-1}} P(M_{i_1},\; M_{i_2},\; \dots,\; M_{i_t})

    and :math:`M_i` is the event that the Multi-Bernoulli is equal to the i-th Multi-Bernoulli of the set.

    Parameters
    ----------
    terms : list of int or None (optinal)
        The terms to consider in the alternated sum. If None, all
        terms are considered (default: None)

    Notes
    -----
        The computation if exponential w.r.t. the number of
        Multi-Bernoulli in the set. This is because the i-th term
        in the alterated sum costs :math:`\binom{l}{i}` where l is the
        number of Multi-Bernoulli in the set.
    """
    def __init__(self, terms=None):
        self.terms = terms
    
    def soft_match(self, x, y):
        """
        Parameters
        ----------
        x : torch.tensor (dtype: float)
            A batch of Multi-Bernolli's logits (pre-sigmoid).
            x.shape = (..., number_of_bits).
        y : torch.tensor (dtype: float)
            A batch of l Multi-Bernoulli's logits (pre-sigmoid).
            y.shape = (..., l, number_of_bits).
            
        Returns
        -------
        log_probs_not_match : torch.tensor (dtype: float).
            log_probs_not_match[i] is the
            log probability that each of the l Multi-Bernoulli
            parametrized by y[i] is different than the
            Multi-Bernoulli parameterized x[i].
        log_probs_match : torch.tensor (dtype: float). 
            log_probs_match[i] is the
            log probability that at least one of the l
            Multi-Bernoulli parametrized by y[i] is equal
            to the Multi-Bernoulli parameterized x[i].
            
        Notes
        -----
        x.shape[:-1] does not broadcast with y.shape[:-2].
            
        Raises
        ------
        ValueError
            If x.ndim + 1 != y.ndim.
        ValueError
            If x.shape[-1] != y.shape[-1] (i.e., different number of bits).
        ValueError
            If no valid odd term (i.e., within 1 up to l) is given in terms.
        """
        if x.ndim + 1 != y.ndim:
            raise ValueError(f'y.ndim must be 1 + x.ndim, got y.ndim={y.ndim} and x.ndim={x.ndim}.')
        
        if x.shape[-1] != y.shape[-1]:
            raise ValueError(f'The number of bits of each Multi-Bernoulli must '
                             f'be the same for both x and y, got {x.shape[-1]} and'
                             f'{y.shape[-1]} respectively.')

        l = y.shape[-2]
        x0, x1, y0, y1 = map(logsigmoid, (-x, x, -y, y))

        terms = range(1, l+1) if self.terms is None else self.terms
        odd_terms = [t for t in terms if t%2==1]
        evn_terms = [t for t in terms if t%2==0]

        odd_log_sum = _membership_log_terms(x0, x1, y0, y1, l, odd_terms)
        evn_log_sum = _membership_log_terms(x0, x1, y0, y1, l, evn_terms)
        if evn_log_sum is None:
            #raises error if odd_log_sum.exp() > 1.
            return torch_logsubexp(torch.zeros_like(odd_log_sum), odd_log_sum), odd_log_sum
        if odd_log_sum is None:
            raise ValueError('At least one valid odd term is required. '
                             'Otherwise, the approximation is negative '
                             'and the logarithm is not defined.')
        evn_log_sum_plus_1 = torch_logsumexp(evn_log_sum, torch.zeros_like(evn_log_sum))
        return torch_logsubexp(evn_log_sum_plus_1, odd_log_sum), torch_logsubexp(odd_log_sum, evn_log_sum)
    
    def hard_match(self, x, y):
        return (~((0 < x)[..., None, :] ^ (0 < y))).all(dim=-1).any(dim=-1)