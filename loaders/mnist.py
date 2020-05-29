import numpy as np
import gzip, pickle
from radbm.loaders.base import Loader, locate_dataset_dirs, locate_dataset_files

class Mnist(Loader):
    def __init__(self, mode, rng=np.random, path=None):
        super().__init__(mode, rng)
        file_paths = locate_dataset_files('Mnist', path, 'mnist.pkl.gz')
        if len(file_paths) == 0: raise FileNotFoundError('could not locate mnist.pkl.gz')
        with gzip.open(file_paths[0], 'rb') as f:
            self.train_xy, self.valid_xy, self.test_xy = pickle.load(f, encoding='latin1')
            self.train_x, self.train_y = self.train_xy
            self.valid_x, self.valid_y = self.valid_xy
            self.test_x, self.test_y = self.test_xy
            
        self.train()
        
    def train(self): self.data_x, self.data_y = self.train_xy; return self
    def valid(self): self.data_x, self.data_y = self.valid_xy; return self
    def test(self): self.data_x, self.data_y = self.test_xy; return self
        
    def batch(self, size=None, index=None, replace=True):
        if index is not None or not replace:
            raise NotImplementedError('not implemented yet')
        ids = self.rng.randint(0, len(self.data_y), size)
        x = self.data_x[ids]
        y = self.data_y[ids]
        return x, y
    
class MnistClass(Mnist):
    def __init__(self, sigma, mode, rng=np.random):
        if mode is not 'Class':
            raise NotImplementedError('Only Class mode is implemented for NoisyMnist')
        super().__init__(mode, rng)
        self.sigma=sigma
        
    def batch(self, size=None, index=None, replace=True):
        x, y = super().batch(size, index, replace)
        q = x + self.rng.normal(0, self.sigma, x.shape)
        d = x + self.rng.normal(0, self.sigma, x.shape)
        return q, d, np.eye(size, dtype=np.uint8)
    
class NoisyMnist(Mnist):
    def __init__(self, sigma, mode, rng=np.random):
        if mode is not 'Relational':
            raise NotImplementedError('Only Relational mode is implemented for NoisyMnist')
        super().__init__(mode, rng)
        self.sigma=sigma
        
    def get_relation_prob(self):
        return 1/len(self.data_x)
    
    def get_relation_log_prob(self):
        return -np.log(len(self.data_x))
        
    def build_any(self, size=None, index=None, replace=True):
        if size is not None or replace:
            raise NotImplementedError('not implemented yet')
        x = self.data_x if index is None else self.data_x[index]
        return x + self.rng.normal(0, self.sigma, x.shape)
    
    def build_queries(self, size=None, index=None, replace=True):
        return self.build_any(size, index, replace) #because the tasks is symmetric
    
    def build_documents(self, size=None, index=None, replace=True):
        return self.build_any(size, index, replace) #because the tasks is symmetric
    
    def batch(self, size=None, index=None, replace=True):
        x, y = super().batch(size, index, replace)
        q = x + self.rng.normal(0, self.sigma, x.shape)
        d = x + self.rng.normal(0, self.sigma, x.shape)
        return q, d, np.eye(size, dtype=np.uint8)