from .base import ModulePointwiseReduction

class HammingReduction(ModulePointwiseReduction):
    def __init__(self, fq, fd, struct):
        super().__init__(struct)
        self.fq = fq
        self.fd = fd
        
    def queries_reduction(self, queries):
        return self.fq(queries) > 0
    
    def documents_reduction(self, documents):
        return self.fd(documents) > 0