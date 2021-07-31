import torch
from radbm.losses import FbetaLoss, BCELoss
from radbm.utils.torch import HuberLoss
from .utils import check_shape_helper, RegularizationMatchingLoss

class MultiBernoulliMatchingLoss(RegularizationMatchingLoss):
    """
    Abstract class, cannot be used directly.
    """
    def __init__(self, log_match, reg=HuberLoss(1, 9), reg_alpha=0):
        super().__init__(reg=reg, reg_alpha=reg_alpha)
        self.log_match = log_match
        
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
        block = check_shape_helper(queries_logits, documents_logits, r)
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