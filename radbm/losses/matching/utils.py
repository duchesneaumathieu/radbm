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