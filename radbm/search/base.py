from radbm.utils.os import StateObj

class BaseSDS(StateObj):
    """
    Base Search Data Structure, need to be instantiate.
    Maintains an index for documents to be retrieved with query.
    When queried the appropriate index(es) will be returned (not
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
        if type(self).batch_insert == BaseSDS.batch_insert:
            raise NotImplementedError('insert or batch_insert need to be overridden')
        self.batch_insert(document[None], [index], *args, **kwargs)
        return self
        
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
        if type(self).insert == BaseSDS.insert:
            raise NotImplementedError('insert or batch_insert need to be overridden')
        for document, index in zip(documents, indexes):
            self.insert(document, index, *args, **kwargs)
        return self
            
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
        if type(self).batch_search == BaseSDS.batch_search:
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
        if type(self).search == BaseSDS.search:
            raise NotImplementedError('search or batch_search need to be overridden')
        return [self.search(q, *args, **kwargs) for q in queries]
    
    def itersearch(self, query, *args, **kwargs):
        """
        Default generator, based on batch_itersearch, that yields indexes.
        
        Parameters
        ----------
            query : numpy.ndarray or torch.Tensor
            *args
                pass to batch_itersearch (see batch_itersearch for more details)
            **kwargs
                pass to batch_itersearch (see batch_itersearch for more details)
            
        Yields
        ------
            indexes
        """
        if type(self).batch_itersearch == BaseSDS.batch_itersearch:
            raise NotImplementedError('itersearch or batch_itersearch need to be overridden')
        yield from self.batch_itersearch(query[None], *args, **kwargs)[0]
        
    def batch_itersearch(self, queries, *args, **kwargs):
        """
        Default function that return a list of generator using itersearch.
        With ith generator corresponding to the ith query.
        
        Parameters
        ----------
            queries : numpy.ndarray or torch.Tensor
            *args
                pass to itersearch (see itersearch for more details)
            **kwargs
                pass to itersearch (see itersearch for more details)
            
        Returns
        ------
            itersearch_list : list of generator
                where the ith generetor yields indexes (set or list) that
                correspond to the ith  query.
        """
        if type(self).itersearch == BaseSDS.itersearch:
            raise NotImplementedError('itersearch or batch_itersearch need to be overridden')
        return [self.itersearch(query, *args, **kwargs) for query in queries]
