from radbm.utils.os import StateObj

class Search(StateObj):
    def batch_insert(self, values, ids):
        for v, i in zip(values, ids):
            self.insert(v, i)
    
    def batch_search(self, values):
        return [self.search(v) for v in values]