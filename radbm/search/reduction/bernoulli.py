import torch
from .base import ModulePointwiseReduction
logsigmoid = torch.nn.LogSigmoid()

def get_log_multi_bernoulli_probs(x):
    return logsigmoid(-x), logsigmoid(x)

class BernoulliReduction(ModulePointwiseReduction):
    def __init__(self, fq, fd, struct):
        super().__init__(struct)
        self.fq = fq
        self.fd = fd
        
    def queries_reduction(self, queries):
        return get_log_multi_bernoulli_probs(self.fq(queries))
    
    def documents_reduction(self, documents):
        return get_log_multi_bernoulli_probs(self.fd(documents))