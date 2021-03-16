import torch
import numpy as np
from radbm.utils.torch import torch_lme, torch_logsumexp

class FbetaLoss(object):
    """
    The F-beta score loss and its variants. This loss is an estimator
    of the log F-beta score of the model.
    
    Parameters
    ----------
    beta : float (positive)
        The parameter that control the importance of the recall over
        the precision.
    prob_y1 : float (in [0, 1])
        The marginal probability that the class is one
    naive : bool (optional)
        If True, the log(recall) of the model is estimated with
        log(mean(recall_hats)). If False mean(log(recall_hats)) 
        is used. (default: False)
    estimator_sharing : bool (optional)
        If True, the log probability of true positive estimators are
        used twice to compute the loss. If False, they are split in 
        two and each halve is used only once. (default: True)
    """
    def __init__(self, beta, prob_y1, naive=False, estimator_sharing=True):
        self.beta = beta
        self.naive = naive
        self.estimator_sharing = estimator_sharing
        self.log_prob_y1 = float(np.log(prob_y1))
        self.log_prob_y0 = float(np.log(1-prob_y1))
        self.log_c1 = float(np.log(prob_y1*beta**2))
        self.log_c3 = float(np.log(1 + 1/beta**2))
    
    def __call__(self, tp_log_probs, fp_log_probs):
        """
        Parameters
        ----------
        tp_log_probs : torch.tensor (dtype: float)
            Estimators of the log probability of a true positive. E.g.
            :math:`log(f_{\\theta}(x))` where :math:`f_{\\theta}` is the model and x is a positive data point.
        fp_log_probs : torch.tensor (dtype: float)
            Estimators of the log probability of a false positive. E.g.
            :math:`log(f_{\\theta}(x))` where :math:`f_{\\theta}` is the model and x is a negative data point.
        """
        ntp = tp_log_probs.flatten().size(0)
        nfp = fp_log_probs.flatten().size(0)
        if ntp == 0 or nfp == 0:
            mgs = 'Fbeta loss requires at least one positive and one negative pair, got {} and {} respectively.'
            raise ValueError(mgs.format(ntp, nfp))
            
        if self.estimator_sharing:
            #doing nothing
            t1_tp_log_probs = t2_tp_log_probs = tp_log_probs
        else:
            #splitting tp_log_probs in two
            tp_log_probs = tp_log_probs.flatten()
            k = len(tp_log_probs)//2
            t1_tp_log_probs = tp_log_probs[:k]
            t2_tp_log_probs = tp_log_probs[k:]
            
        if self.naive:
            #computing the log mean exp
            term1 = torch_lme(t1_tp_log_probs)
        else:
            #simply the mean of the tp_log_probs
            term1 = t1_tp_log_probs.mean()
            
        log_recall = torch_lme(t2_tp_log_probs)
        log_fallout = torch_lme(fp_log_probs)
        log_t = torch_logsumexp(
            self.log_prob_y1 + log_recall,          
            self.log_prob_y0 + log_fallout)
        term2 = torch.nn.Softplus()(log_t - self.log_c1)
        log_fbeta_score_estimator =  term1 - term2 + self.log_c3
        return -log_fbeta_score_estimator #for minimization
    
class BCELoss(object):
    """
    Similar to torch.nn.BCELoss, but where the weighting is class-based
    instead of data point-based. E.g. If y in {0,1}^n are classes and
    p \in [0, 1]^n are the model estimated probabilities then
    torch.nn.BCELoss(weight)(p, y) = BCELoss(w1)(log(p)[y], log(1-p)[~y])
    where weight = (~y)*(2-w1) + y*w1.
    
    Parameters
    ----------
    w1 : float (in [0, 2])
        The weight for the positive class. The weight for the negative
        class is 2 - w1.
    """
    def __init__(self, w1):
        self.w1 = w1
        self.w0 = 2 - w1
    
    def __call__(self, tp_log_probs, tn_log_probs):
        """
        Parameters
        ----------
        tp_log_probs : torch.tensor (dtype: float)
            Estimators of the log probability of a true positive. E.g.
            :math:`log(f_{\\theta}(x))` where :math:`f_{\\theta}` is the model and x is a positive data point.
        tn_log_probs : torch.tensor (dtype: float)
            Estimators of the log probability of a true negative. E.g.
            :math:`log(1-f_{\\theta}(x))` where :math:`f_{\\theta}` is the model and x is a negative data point.
        """
        ntp = tp_log_probs.flatten().size(0)
        ntn = tn_log_probs.flatten().size(0)
        if ntp == 0 or ntn == 0:
            mgs = 'BCE loss requires at least one positive and one negative pair, got {} and {} respectively.'
            raise ValueError(mgs.format(ntp, ntn))
            
        n = len(tp_log_probs.flatten()) + len(tn_log_probs.flatten())
        s = self.w1*tp_log_probs.sum() + self.w0*tn_log_probs.sum()
        return -s/n