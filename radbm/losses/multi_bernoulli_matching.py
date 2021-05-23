import torch
from radbm.losses import FbetaLoss, BCELoss
from radbm.utils.torch import HuberLoss

def check_shape_helper(queries_logits, documents_logits, r, block):
    qbs = queries_logits.shape[0]
    dbs = documents_logits.shape[0]
    if block:
        if r.shape[0] != qbs:
            msg = 'r.shape[0] != queries_logits[0], got {} and {} respectively.'
            raise ValueError(msg.format(r.shape[0], qbs))
        if r.shape[1] != dbs:
            msg = 'r.shape[1] != documents_logits[0], got {} and {} respectively.'
            raise ValueError(msg.format(r.shape[1], dbs))
    else:
        if not r.shape[0]==qbs==dbs:
            msg = 'If not block (r.ndim==1), the number of queries and documents must be the same, got {} and {} respectively.'
            raise ValueError(msg.format(qbs, dbs))

class MultiBernoulliMatchingLoss(object):
    """
    Abstract class, cannot be used directly.
    """
    def __init__(self, log_match, reg=HuberLoss(1, 9), reg_alpha=0):
        self.log_match = log_match
        self.reg = reg
        self.reg_alpha = reg_alpha
    
    def regularization(self, loss, queries_logits, documents_logits):
        if self.reg_alpha:
            n = queries_logits.flatten().size(0) + documents_logits.flatten().size(0)
            reg = (self.reg(queries_logits).sum() + self.reg(documents_logits).sum())/n
            loss = loss + self.reg_alpha*reg
        return loss
        
    def get_log_probs(self, queries_logits, documents_logits, r):
        """
        Parameters
        ----------
        queries_logits : torch.tensor (dtype: float)
            Last dim must be w.r.t. the bits. If ndim must be 2 or 3. If ndim==3,
            the second dim must correspond to the number of indexes.
        documents_logits : torch.tensor (dtype: float)
            Last dim must be w.r.t. the bits. If ndim must be 2 or 3. If ndim==3,
            the second dim must correspond to the number of indexes.
        r : torch.tensor (dtype: bool)
            Whether the corresponding query-document pair match or not. ndim must be 1 or 2.
            If ndim==1, r[i] indicates if queries_logits[i] match with documents_logits[i].
            If ndim==2, r[i,j] indicates if queries_logits[i] match with documents_logits[j].
            
        Raises
        ------
        ValueError
            If r.ndim==1, r.shape[0] must be equal to queries_logits.shape[0] and documents_logits.shape[0].
            If r.ndim==2, r.shape[0] must be equal to queries_logits.shape[0] and r.shape[1] must be equal to
            documents_logits.shape[0]. Otherwise a ValueError will be raised.
            
        Returns
        -------
        loss : torch.tensor (dtype: float, shape: ())
        """
        block = (r.ndim == 2)
        check_shape_helper(queries_logits, documents_logits, r, block)
        if block:
            queries_logits = queries_logits.unsqueeze(1)
            documents_logits = documents_logits.unsqueeze(0)
        return self.log_match(queries_logits, documents_logits)
        
class FbetaMultiBernoulliMatchingLoss(MultiBernoulliMatchingLoss):
    """
    Callable, see MultiBernoulliMatchingLoss for documentation of how to call this.
    
    Parameters
    ----------
    log_match : function
        A matching function which takes logits and output log probabilities.
        E.g. HammingMatch(dist=0).
    log2_beta : float
        See FbetaLoss.
    prob_y1 : float
        See FbetaLoss.
    naive : bool
        See FbetaLoss
    estimator_sharing : bool
        See FbetaLoss
    reg : function (optional)
        The regularization over the logits to use. (default HuberLoss(1, 9))
    reg_alpha : float (optional)
        The factor to multiply the regularization with before adding it to the loss.
        (default 0., i.e. deactivated regularization)
    """
    def __init__(self, log_match, log2_beta, prob_y1, naive=False, estimator_sharing=True, reg=HuberLoss(1, 9), reg_alpha=0.,):
        super().__init__(log_match, reg=reg, reg_alpha=reg_alpha)
        self.loss = FbetaLoss(
            log2_beta=log2_beta,
            prob_y1=prob_y1,
            naive=naive,
            estimator_sharing=estimator_sharing,
        )
    
    def __call__(self, queries_logits, documents_logits, r, step=None):
        neg_log_probs, pos_log_probs = self.get_log_probs(queries_logits, documents_logits, r)
        
        not_r = ~r
        tp_log_probs = pos_log_probs[r]
        fp_log_probs = pos_log_probs[not_r]
        #F-beta loss does not need neg_log_probs.
        
        loss = self.loss(tp_log_probs, fp_log_probs, step=step)
        return self.regularization(loss, queries_logits, documents_logits)
    
class BCEMultiBernoulliMatchingLoss(MultiBernoulliMatchingLoss):
    """
    Callable, see MultiBernoulliMatchingLoss for documentation of how to call this.
    
    Parameters
    ----------
    log_match : function
        A matching function which takes logits and output log probabilities.
        E.g. HammingMatch(dist=0).
    log2_lambda : float or tuple (optional)
        See radbm.losses.BCELoss. (default: -1., i.e. usual BCE)
    reg : function (optional)
        The regularization over the logits to use. (default HuberLoss(1, 9))
    reg_alpha : float (optional)
        The factor to multiply the regularization with before adding it to the loss.
        (default 0., i.e. deactivated regularization)
    """
    def __init__(self, log_match, log2_lambda=-1., reg=HuberLoss(1, 9), reg_alpha=0.):
        super().__init__(log_match, reg=reg, reg_alpha=reg_alpha)
        self.loss = BCELoss(log2_lambda=log2_lambda)
    
    def __call__(self, queries_logits, documents_logits, r, step=None):
        neg_log_probs, pos_log_probs = self.get_log_probs(queries_logits, documents_logits, r)
        
        not_r = ~r
        tp_log_probs = pos_log_probs[r]
        tn_log_probs = neg_log_probs[not_r] #difference with Fbeta
        
        loss = self.loss(tp_log_probs, tn_log_probs, step=step)
        return self.regularization(loss, queries_logits, documents_logits)