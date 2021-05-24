from radbm.search.base import BaseSDS

class DictionarySearch(dict, BaseSDS):
    def insert(self, document, index):
        if document in self: self[document].add(index)
        else: self[document] = {index}
        return self
    
    def search(self, query):
        if query in self: return self[query]
        else: return set()
        
    def get_state(self):
        """
        Returns
        -------
        table : dict
            The current hash table
        """
        return self
    
    def set_state(self, state):
        """
        Parameters
        -------
        state : dict
            The hash table to use
        """
        self.clear()
        self.update(state)
        return self