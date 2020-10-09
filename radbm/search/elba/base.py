import torch
from radbm.search.base import BaseSDS
logsigmoid = torch.nn.LogSigmoid()

class EfficientLearnableBinaryAccess(BaseSDS, torch.nn.Module):
    """
    EfficientLearnableBinaryAccess (ELBA) is a base class for concrete models.
    Given a search data structure and two parametric encoding functions, one
    for the queries (fq) and one for the documents (fd), both producing a 
    Multi-Bernoulli code (i.e. in [0,1]^n) in its logits form (pre sigmoid).
    ELBM uses the structure to store and retrieve data using the Multi-Bernoulli
    code.
    
    Parameters
    ----------
    fq : torch.nn.Module
        The parametric function of the queries outputting in logits (pre sigmoid).
    fd : torch.nn.Module
        The parametric function of the documents outputting in logits (pre sigmoid).
    struct : BaseSDS subclass
        The structure used for storing and retrieval.
    optim : torch.optim (optional)
        The optimizer (minimizer) class used for the parametric function.
        (default torch.optim.Adam)
    lr : float (optional)
        The learning rate of the optimizer. (default 0.001)
    """
    def __init__(self, fq, fd, struct, optim=torch.optim.Adam, lr=0.001):
        torch.nn.Module.__init__(self)
        self.fq = fq
        self.fd = fd
        self.struct = struct
        self.optim = optim(self.parameters(), lr)
        
    def _log_sigmoid_pairs(self, logits):
        return torch.stack([logsigmoid(-logits), logsigmoid(logits)], dim=1)
        
    def batch_insert(self, documents, indexes, *args, **kwargs):
        """
        Insert the index of each documents in the data structure
        
        Parameters
        ----------
        documents : torch.tensor
            The documents to insert the first dim being the batch.
        indexes : iterable of hashable
            most notable example is a list of int. len(indexes) most
            be equal to len(documents).
        *args
            passed to self.struct.batch_insert
        **kwargs
            passed to self.struct.batch_insert
            
        Returns
        -------
        self
        """
        dmb = self._log_sigmoid_pairs(self.fd(documents))
        self.struct.batch_insert(dmb, indexes, *args, **kwargs)
        return self
        
    def batch_search(self, queries, *args, **kwargs):
        """
        Search in the data structure for the relevant indexes for each queries.
        
        Parameters
        ----------
        queries : torch.tensor
            The search queries, the first dim being the batch.
        *args
            passed to self.struct.batch_search
        **kwargs
            passed to self.struct.batch_search
            
        Returns
        -------
        indexes_list : list of (set or list)
            Is the list of the relevant indexes for each queries. 
            len(indexes_list) = len(queries).
        """
        qmb = self._log_sigmoid_pairs(self.fq(queries))
        return self.struct.batch_search(qmb, *args, **kwargs)
        
    def batch_itersearch(self, queries, *args, **kwargs):
        """
        Iteratively search in the data structure for the relevant
        indexes for each queries.
        
        Parameters
        ----------
        queries : torch.tensor
            The search queries, the first dim being the batch.
        *args
            passed to self.struct.batch_itersearch
        **kwargs
            passed to self.struct.batch_itersearch
            
        Returns
        -------
        generator_list : list of generators (of set or list)
            Each generator yield relevant indexes for the corresponding queries. 
            len(generator_list) = len(queries).
        """
        qmb = self._log_sigmoid_pairs(self.fq(queries))
        return self.struct.batch_itersearch(qmb, *args, **kwargs)
    
    def get_state(self):
        return {
            'f': self.state_dict(),
            'optim': self.optim.state_dict(),
            'struct': self.struct.get_state(),
        }
    
    def set_state(self, state):
        self.load_state_dict(state['f'])
        self.optim.load_state_dict(state['optim'])
        self.struct.set_state(state['struct'])
        return self