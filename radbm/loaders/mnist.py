import numpy as np
import gzip, pickle, torch
from radbm.loaders.base import Loader, IRLoader
from radbm.utils.fetch import get_directories_list, fetch_file

def mnist_loader(path=None, download=True):
    file_paths = fetch_file('mnist.pkl.gz', path, data_type='dataset', subdirs=['Mnist', 'mnist'], download=download)
    if not file_paths:
        raise FileNotFoundError('could not locate mnist.pkl.gz')
    with gzip.open(file_paths[0], 'rb') as f:
        train_xy, valid_xy, test_xy = pickle.load(f, encoding='latin1')
    return train_xy, valid_xy, test_xy
    
class Mnist(Loader):
    def __init__(self, path=None, download=True, which='train', backend='numpy', device='cpu', rng=np.random):
        super().__init__(which=which, backend=backend, device=device, rng=rng)
        train_xy, valid_xy, test_xy = mnist_loader(path, download=download)
        for which, (x, y) in [('train', train_xy), ('valid', valid_xy), ('test', train_xy)]:
            self.register_switch('{}_x'.format(which), x)
            self.register_switch('{}_y'.format(which), y)
            x = getattr(self, '{}_x'.format(which)) #retrieve the TorchNumpy
            y = getattr(self, '{}_y'.format(which)) #retrieve the TorchNumpy
            setattr(self, '{}_group'.format(which), {'x':x, 'y':y})
        getattr(self, self.which)()
        
    def batch(self, size=None, index=None, replace=True):
        if index is not None or not replace:
            raise NotImplementedError('not implemented yet')
        ids = self.rng.randint(0, len(self.y.data), size)
        return self.x.data[ids], self.y.data[ids]
    
class ClassMnist(IRLoader):
    def __init__(self, sigma=0, path=None, download=True, mode='class_relation', which='train', backend='numpy', device='cpu', rng=np.random):
        super().__init__(mode=mode, which=which, backend=backend, device=device, rng=rng)
        if mode is not 'relational_matrix':
            raise NotImplementedError('mode={} not implemented yet'.format(mode))
        self.sigma = sigma
        train_xy, valid_xy, test_xy = mnist_loader(path, download=download)
        for which, (x, y) in [('train', train_xy), ('valid', valid_xy), ('test', train_xy)]:
            self.register_switch('{}_x'.format(which), x[y.argsort()])
            jump = np.arange(0,len(x),len(x)//10) #jump always on np since it is for indexing
            x = getattr(self, '{}_x'.format(which)) #retrieve the TorchNumpy
            setattr(self, '{}_group'.format(which), {'x':x, 'jump':jump})
        self.register_switch('eye', np.eye(10, dtype=np.bool))
        getattr(self, self.which)()
        
    def get_available_modes(self):
        return {
            'class_relation',
            'relational_triplets',
            'relational_matrix'
        }
        
    def get_relation_prob(self):
        return 1/10
    
    def get_relation_log_prob(self):
        return -np.log(10)
        
    def build_any(self, size=None, index=None, replace=True):
        if size is not None or replace:
            raise NotImplementedError('not implemented yet')
        x = self.x.data if index is None else self.x.data[index]
        return x + self.rng.normal(0, self.sigma, x.shape)
    
    def build_queries(self, size=None, index=None, replace=True):
        return self.build_any(size, index, replace) #because the tasks is symmetric
    
    def build_documents(self, size=None, index=None, replace=True):
        return self.build_any(size, index, replace) #because the tasks is symmetric
    
    def batch(self, size=None, index=None, replace=True):
        if size != 10: raise NotImplementedError('batch size != 10 not supported')
        x = self.x.data
        ids1 = self.rng.rng.randint(0, len(x)//10, size) + self.jump
        ids2 = self.rng.rng.randint(0, len(x)//10, size) + self.jump
        x1, x2 = x[ids1], x[ids2]
        x1 = x1 + self.rng.normal(0, self.sigma, x1.shape)
        x2 = x2 + self.rng.normal(0, self.sigma, x2.shape)
        return x1, x2, self.eye.data
    
class NoisyMnist(IRLoader):
    def __init__(self, sigma=0.2, path=None, download=True, mode='relational_matrix', which='train', backend='numpy', device='cpu', rng=np.random):
        super().__init__(mode=mode, which=which, backend=backend, device=device, rng=rng)
        if mode is not 'relational_matrix':
            raise NotImplementedError('mode={} not implemented yet'.format(mode))
        self.sigma = sigma
        train_xy, valid_xy, test_xy = mnist_loader(path, download=download)
        for which, (x, y) in [('train', train_xy), ('valid', valid_xy), ('test', train_xy)]:
            self.register_switch('{}_x'.format(which), x)
            x = getattr(self, '{}_x'.format(which)) #retrieve the TorchNumpy
            setattr(self, '{}_group'.format(which), {'x':x})
        getattr(self, self.which)()
    
    def get_available_modes(self):
        return {
            'relational_pairs', #(x,k,r)
            'relational_triplets', #(x, pos_x, neg_x)
            'relational_matrix', #(xs, ks, R)
        }
    
    def get_relation_prob(self):
        return 1/len(self.x.data)
    
    def get_relation_log_prob(self):
        return -np.log(len(self.x.data))
        
    def build_any(self, size=None, index=None, replace=True):
        if size is not None or replace:
            raise NotImplementedError('not implemented yet')
        x = self.x.data if index is None else self.x.data[index]
        return x + self.rng.normal(0, self.sigma, x.shape)
    
    def build_queries(self, size=None, index=None, replace=True):
        return self.build_any(size, index, replace) #because the tasks is symmetric
    
    def build_documents(self, size=None, index=None, replace=True):
        return self.build_any(size, index, replace) #because the tasks is symmetric
    
    def batch(self, size=None, index=None, replace=True):
        if index is not None or not replace:
            raise ValueError('not implemented yet')
        if size is None: size = len(self.x.data)
        ids = self.rng.rng.randint(0, len(self.x.data), size)
        x = self.x.data[ids]
        q = x + self.rng.normal(0, self.sigma, x.shape)
        d = x + self.rng.normal(0, self.sigma, x.shape)
        r = self.dynamic_cast(np.eye(size, dtype=np.bool))
        return q, d, r