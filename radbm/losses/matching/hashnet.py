import torch
from .utils import check_shape_helper
softplus = torch.nn.Softplus()

class HashNetMatchingLoss(object):
    r"""
    As in `HashNet: Deep Learning to Hash by Continuation  <https://arxiv.org/abs/1702.00758>`__.

    Parameters
    ----------
    match_prob : float (in [0,1])
        The probability that there is a match given a random query
        and a random document, used for class balancing.
    alpha : float (optional)
        The HashNet's alpha used in the adaptative sigmoid, there is no recommendation in the article.
        Maybe 6/nbits is a good place to start. (default: 1)
    beta0 : float (optional)
        The initial HashNet's beta. Beta increase linearly with the stage, i.e. :math:`\mathrc{tanh}(\mathrc{stage} \beta_0 x)`.
        (default: 1)
    stage_length : int (optional)
        The number of steps before increasing the stage. (default: 10000)
    """
    def __init__(self, log2_lambda=-1, alpha=1, beta0=1, stage_length=10000):
        self.log2_lambda = log2_lambda
        self.alpha = alpha
        self.beta0 = beta0
        self.stage_length = stage_length
    
    def _get_weight(self, r):
        lmda = 2**self.log2_lambda
        n = len(r); n1 = r.sum(); n0 = n - n1
        w1 = n*lmda/n1
        w0 = (n-n1*w1)/n0
        return w1*r + w0*~r
    
    def __call__(self, queries_logits, documents_logits, r, step=0):
        """
        Parameters
        ----------
        queries_logits : torch.Tensor (ndim: 2)
            A batch of queries.
        documents_logits : torch.Tensor (ndim: 2)
            A batch of documents.
        r : torch.Tensor (dtype: torch.bool, ndim: 1 or 2)
            A matrix (block mode) (2D tensor) with r[i,j] indicating if q[i] match with d[j] or a vector (1D tensor) with
            r[i] indicating if q[i] match with d[i]
        stage : float (optional)
            The stage for computing the tanh(stage*beta*x). (default=1)
            
        Returns
        -------
        loss : torch.Tensor (size 1)
            The loss of HashNet.
        """
        if queries_logits.ndim != 2:
            msg = f'queries_logits.ndim must be 2, got {queries_logits.ndim}.'
            raise ValueError(msg)
        if documents_logits.ndim != 2:
            msg = f'documents_logits.ndim must be 2, got {documents_logits.ndim}.'
            raise ValueError(msg)
        if r.ndim not in {1, 2}:
            msg = f'r.ndim must be 1 (normal) or 2 (block), got {r.ndim}.'
            raise ValueError(msg)
        if r.dtype != torch.bool:
            msg = f'relevants must be boolean, got {r.dtype}.'
            raise TypeError(msg)
            
        block = check_shape_helper(queries_logits, documents_logits, r)
        if block:
            queries_logits = queries_logits.unsqueeze(1)
            documents_logits = documents_logits.unsqueeze(0)
            
        stage = int(step//self.stage_length) + 1
        bq = torch.tanh(stage*self.beta0*queries_logits)
        bd = torch.tanh(stage*self.beta0*documents_logits)
        sh = (bq*bd).sum(dim=-1)
        ash = self.alpha*sh
        w = self._get_weight(r)
        losses = w*(softplus(ash) - r*ash)
        loss = losses.mean()
        return loss