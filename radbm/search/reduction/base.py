import torch
from radbm.search.base import BaseSDS

class PointwiseReduction(BaseSDS):
    """
    Abstract class, queries_reduction and documents_reduction methods need to be overwritten.
    A Pointwise reduction is the simplest form of reduction. Each document is transformed without taking
    the other documents in consideration. Similarly, each query is transformed without looking at the database.
    
    Parameters
    ----------
    struct : BaseSDS subclass
        The data structure to reduce to.
    """
    def __init__(self, struct):
        self.struct = struct
    
    def queries_reduction(self, queries):
        raise NotImplementedError()
        
    def documents_reduction(self, documents):
        raise NotImplementedError()
        
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
        reduced_documents = self.documents_reduction(documents)
        self.struct.batch_insert(reduced_documents, indexes, *args, **kwargs)
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
        reduced_queries = self.queries_reduction(queries)
        return self.struct.batch_search(reduced_queries, *args, **kwargs)
        
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
        reduced_queries = self.queries_reduction(queries)
        return self.struct.batch_itersearch(reduced_queries, *args, **kwargs)
    
    def clear(self):
        self.struct.clear()
        
class ModulePointwiseReduction(PointwiseReduction, torch.nn.Module):
    """
    Abstract class, queries_reduction and documents_reduction methods need to be overwritten.
    A Pointwise reduction is the simplest form of reduction. Each document is transformed without taking
    the other documents in consideration. Similarly, each query is transformed without looking at the database.
    
    Direct subclass of PointwiseReduction that implement get_state and set_state. Allowing the save and load methods.
    This is done in a way that any attribute of the class torch.nn.Module will be saved.
    
    Parameters
    ----------
    struct : BaseSDS subclass
        The data structure to reduce to.
    """
    def __init__(self, struct):
        super().__init__(struct)
        torch.nn.Module.__init__(self)
        
    def get_state(self):
        return {
            'module': self.state_dict(),
            'struct': self.struct.get_state(),
        }
    
    def set_state(self, state):
        self.load_state_dict(state['module'])
        self.struct.set_state(state['struct'])
        return self