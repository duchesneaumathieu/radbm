from radbm.utils.os import StateObj

class Retrieval(StateObj):
    """
    Maintains an index for documents to be retrieved with query
    when queried the appropriate index(es) will be returned (not
    the document(s))
    """
    def insert(self, document, index, *args, **kwargs):
        """
        Insert a single document in the data structure by saving the
        index. The document is not saved for scalability. This is
        the default implementation that uses batch_insert. See batch_insert
        documentation.
        
        Parameters
        ----------
            document : numpy.ndarray or torch.Tensor
                The document to by inserted (will not be saved)
            index : object
                A unique identifier for the document, some algorithms
                require index to be hashable.
            *args
                pass to batch_insert (see batch_insert for more details)
            **kwargs
                pass to batch_insert (see batch_insert for more details)
                
        """
        if type(self).batch_insert == Retrieval.batch_insert:
            raise NotImplementedError('insert or batch_insert need to be overridden')
        self.batch_insert(document[None], [index], *args, **kwargs)
        
    def batch_insert(self, documents, indexes, *args, **kwargs):
        """
        Insert a multiple documents in the data structure by saving their
        index. The documents are not saved for scalability. This is
        the default implementation that uses insert. See insert
        documentation.
        
        Parameters
        ----------
            documents : numpy.ndarray or torch.Tensor
                The documents to by inserted (will not be saved). The first
                dimension is for the batch (i.e. documents[0] is a document)
            indexes : object
                A unique identifier for the document, some algorithms
                require index to be hashable.
            *args
                pass to insert (see insert for more details)
            **kwargs
                pass to insert (see insert for more details)
        """
        if type(self).insert == Retrieval.insert:
            raise NotImplementedError('insert or batch_insert need to be overridden')
        for document, index in zip(documents, indexes):
            self.insert(document, index, *args, **kwargs)
            
    def search(self, query, *args, **kwargs):
        """
        Search in the data structure for the index of a documents. This is
        the default implementation that uses batch_search. See batch_search
        documentation.
        
        Parameters
        ----------
            query : numpy.ndarray or torch.Tensor
            *args
                pass to batch_search (see batch_search for more details)
            **kwargs
                pass to batch_search (see batch_search for more details)
            
        Returns
        -------
            indexes : set or list
                The indexes of the retrieved documents. If indexes
                is a list, it should indicate that the indexes are ordered.
        """
        if type(self).batch_search == Retrieval.batch_search:
            raise NotImplementedError('search or batch_search need to be overridden')
        return self.batch_search(query[None], *args, **kwargs)[0]
    
    def batch_search(self, queries, *args, **kwargs):
        """
        Search in the data structure for the index of a documents. This is
        the default implementation that uses search. See search
        documentation.
        
        Parameters
        ----------
            query : numpy.ndarray or torch.Tensor
            *args
                pass to search (see search for more details)
            **kwargs
                pass to search (see search for more details)
            
        Returns
        -------
            indexes_list : list of (set or list)
                The indexes of the retrieved documents for each query.
                If indexes[i] is a list, it should indicate that the
                indexes are ordered.
        """
        if type(self).search == Retrieval.search:
            raise NotImplementedError('search or batch_search need to be overridden')
        return [self.search(q, *args, **kwargs) for q in queries]
    
    def itersearch(self, query):
        """
        itersearch should be overridden. This should be a generator
        of the index.
        
        Parameters
        ----------
            query : numpy.ndarray or torch.Tensor
            
        Yields
        ------
            index
        """
        raise NotImplementedError('itersearch need to be overridden')
        
    def batch_itersearch(self, query):
        """
        itersearch should be overridden. This should be a generator
        of the batch of index. This should not be confused with the other
        batch_* methods as this take only a single query (not a batch of queries)
        and generate batch of indexes.
        
        Parameters
        ----------
            query : numpy.ndarray or torch.Tensor
            
        Yields
        ------
            batch_index : set or list of index
                A list should indicate that the batch is ordered.
        """
        raise NotImplementedError('batch_itersearch need to be overridden')