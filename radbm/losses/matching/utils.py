def check_shape_helper(queries_logits, documents_logits, r):
    block = (r.ndim == 2)
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
    return block

class RegularizationMatchingLoss(object):
    """
    Abstract class, cannot be used directly.
    """
    def __init__(self, reg, reg_alpha):
        self.reg = reg
        self.reg_alpha = reg_alpha
    
    def regularization(self, loss, queries_logits, documents_logits):
        if self.reg_alpha:
            n = queries_logits.flatten().size(0) + documents_logits.flatten().size(0)
            reg = (self.reg(queries_logits).sum() + self.reg(documents_logits).sum())/n
            loss = loss + self.reg_alpha*reg
        return loss