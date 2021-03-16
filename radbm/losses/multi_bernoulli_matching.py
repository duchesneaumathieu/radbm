import torch
from radbm.losses import FbetaLoss, BCELoss
from radbm.utils.torch import torch_log_prob_any, positive_loss_adaptative_l2_reg

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

def view_helper(queries_logits, documents_logits, block, multi):
    qsh = queries_logits.shape
    dsh = documents_logits.shape
    if block:
        q_frontshape = (qsh[0], 1)
        d_frontshape = (1, dsh[0])
    else:
        q_frontshape = qsh[:1]
        d_frontshape = dsh[:1]
        
    if multi:
        q_ncodes = 1 if queries_logits.ndim==2 else qsh[1]
        d_ncodes = 1 if documents_logits.ndim==2 else dsh[1]
        q_backshape = (q_ncodes, 1, qsh[-1])
        d_backshape = (1, d_ncodes, dsh[-1])
    else:
        q_backshape = (qsh[-1],)
        d_backshape = (dsh[-1],)
        
    q_view = queries_logits.view(q_frontshape + q_backshape)
    d_view = documents_logits.view(d_frontshape + d_backshape)
    return q_view, d_view

def log_probs_helper(queries_logits, documents_logits, block, log_match):
    multi = (queries_logits.ndim == 3 or documents_logits.ndim == 3)
    queries_logits, documents_logits = view_helper(queries_logits, documents_logits, block, multi)
    pos_log_probs, neg_log_probs = log_match(queries_logits, documents_logits)
    if multi:
        if block:
            #*_log_probs.shape = (qsh[0], dsh[0], q_ncodes, d_ncodes)
            #-> *_log_probs.shape = (qsh[0], dsh[0], ncodes)
            frontshape = pos_log_probs.shape[:2]
        else:
            #*_log_probs.shape = (bs, q_ncodes, d_ncodes)
            #-> *_log_probs.shape = (bs, ncodes)
            frontshape = pos_log_probs.shape[:1]
        ncodes = pos_log_probs.shape[-2] * pos_log_probs.shape[-1]
        pos_log_probs = pos_log_probs.view(frontshape + (ncodes,))
        neg_log_probs = neg_log_probs.view(frontshape + (ncodes,))
        pos_log_probs, neg_log_probs = torch_log_prob_any(pos_log_probs, neg_log_probs)
    return pos_log_probs, neg_log_probs

class MultiBernoulliMatchingLoss(object):
    """
    Abstract class, cannot use directly.
    """
    def __init__(self, log_match, l2_ratio=0, *args, **kwargs):
        self.log_match = log_match
        self.l2_ratio = l2_ratio
        self.loss = self._build_loss(*args, **kwargs)
        
    def __call__(self, queries_logits, documents_logits, r):
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
        pos_log_probs, neg_log_probs = log_probs_helper(queries_logits, documents_logits, block, self.log_match)
        loss = self.loss(*self._loss_inputs(pos_log_probs, neg_log_probs, r))
        if self.l2_ratio:
            loss = positive_loss_adaptative_l2_reg(loss, self.l2_ratio, [queries_logits, documents_logits])
        return loss
    
class FbetaMultiBernoulliMatchingLoss(MultiBernoulliMatchingLoss):
    """
    Callable, see MultiBernoulliMatchingLoss for documentation of how to call this.
    
    Parameters
    ----------
    log_match : function
        A matching function which takes logits and output log probabilities.
        E.g. HammingMatch(dist=0).
    l2_ratio : float
        Adaptive l2 regularization, the higher the loss the higher the regularization.
        The final loss is loss + alpha*L2 and alpha is choosen s.t.
        l2_ratio*loss = alpha*L2. (the gradient does not go through alpha)
    beta : float
        See FbetaLoss.
    prob_y1 : float
        See FbetaLoss.
    """
    def _build_loss(self, *args, **kwargs):
        return FbetaLoss(*args, **kwargs)
    
    def _loss_inputs(self, pos_log_probs, neg_log_probs, r):
        not_r = ~r
        tp_log_probs = pos_log_probs[r]
        fp_log_probs = pos_log_probs[not_r]
        return tp_log_probs, fp_log_probs
        
class BCEMultiBernoulliMatchingLoss(MultiBernoulliMatchingLoss):
    """
    Callable, see MultiBernoulliMatchingLoss for documentation of how to call this.
    
    Parameters
    ----------
    log_match : function
        A matching function which takes logits and output log probabilities.
        E.g. HammingMatch(dist=0).
    l2_ratio : float
        Adaptive l2 regularization, the higher the loss the higher the regularization.
        The final loss is loss + alpha*L2 and alpha is choosen s.t.
        l2_ratio*loss = alpha*L2. (the gradient does not go through alpha)
    w1 : float
        See radbm.losses.BCELoss.
    """
    def _build_loss(self, *args, **kwargs):
        return BCELoss(*args, **kwargs)
    
    def _loss_inputs(self, pos_log_probs, neg_log_probs, r):
        not_r = ~r
        tp_log_probs = pos_log_probs[r]
        tn_log_probs = neg_log_probs[not_r]
        return tp_log_probs, tn_log_probs