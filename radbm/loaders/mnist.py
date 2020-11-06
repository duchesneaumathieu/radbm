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
        if mode != 'relational_matrix':
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
        if mode != 'relational_matrix':
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
    
    def iter_any(self, batch_size, seed, maximum):
        which = self.which
        N = min(maximum, len(self.x.data))
        nbatch = int(np.ceil(N//batch_size))
        rng = np.random.RandomState(seed)
        for start in [i*batch_size for i in range(nbatch)]:
            if self.which != which:
                msg = 'self.which changed while iterating, should be "{}" but is now "{}".'
                raise RuntimeError(msg.format(which, self.which))
            end = min(start+batch_size, N)
            data = self.x.data[start:end].reshape(-1, 28, 28)
            data = data + self.dynamic_cast(rng.normal(0, self.sigma, data.shape))
            yield data, list(range(start, end))
            
    def iter_queries(self, batch_size, maximum=np.inf):
        for data, ids in self.iter_any(batch_size, seed=0xcafe, maximum=maximum):
            yield data, [{i} for i in ids]
    
    def iter_documents(self, batch_size, maximum=np.inf):
        yield from self.iter_any(batch_size, seed=0xbeef, maximum=maximum)
    
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
    
def _reverse_index(index):
    r = dict()
    for n, i in enumerate(index):
        for j in i.flatten():
            if j in r: r[j].add(n)
            else: r[j] = {n}
    return [r[i] for i in range(10000)]

class MnistMosaic(IRLoader):
    def __init__(self, sigma=0.2, path=None, download=True, mode='relational_matrix', which='train', backend='numpy', device='cpu', rng=np.random):
        super().__init__(mode=mode, which=which, backend=backend, device=device, rng=rng)
        if mode != 'relational_matrix':
            raise NotImplementedError('mode={} not implemented yet'.format(mode))
        self.sigma = sigma
        dbrng = np.random.RandomState(0xcafe) #used only for indexes generation
        train_xy, valid_xy, test_xy = mnist_loader(path, download=download)
        for which, (x, y) in [('train', train_xy), ('valid', valid_xy), ('test', train_xy)]:
            self.register_switch('{}_x'.format(which), x)
            x = getattr(self, '{}_x'.format(which)) #retrieve the TorchNumpy
            documents_indexes = dbrng.randint(0, 10000, (100000, 2, 2))
            axis1, axis2 = np.random.randint(0,2,(2, 10000))
            documents_indexes[range(10000), axis1, axis2] = range(10000) #so the n query is relevant the the n document
            relevants = _reverse_index(documents_indexes) #might crash with "bad" generator seed
            setattr(self, '{}_group'.format(which), {
                'x' :x,
                'documents_indexes': documents_indexes,
                'relevants': relevants,
            })
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
                    
    def dynamic_backend_mosaic(self, ids):
        x = self.x.data[ids.flatten()]
        if self.backend == 'torch':
            x = x.view(-1, 2, 2, 28, 28).permute(0,1,3,2,4).reshape(-1, 56, 56)
        elif self.backend == 'numpy':
            x = x.reshape(-1, 2, 2, 28, 28).transpose(0,1,3,2,4).reshape(-1, 56, 56)
        return x
    
    def iter_queries(self, batch_size, maximum=np.inf):
        which = self.which
        N = min(maximum, len(self.x.data))
        nbatch = int(np.ceil(N//batch_size))
        rng = np.random.RandomState(0xcafe)
        for start in [i*batch_size for i in range(nbatch)]:
            if self.which != which:
                msg = 'self.which changed while iterating, should be "{}" but is now "{}".'
                raise RuntimeError(msg.format(which, self.which))
            end = min(start+batch_size, N)
            data = self.x.data[start:end].reshape(-1, 28, 28)
            data = data + self.dynamic_cast(rng.normal(0, self.sigma, data.shape))
            relevants = self.relevants[start:end]
            yield data, relevants
    
    def iter_documents(self, batch_size, maximum=np.inf):
        which = self.which
        N = min(maximum, len(self.documents_indexes))
        nbatch = int(np.ceil(N//batch_size))
        rng = np.random.RandomState(0xbeef)
        for start in [i*batch_size for i in range(nbatch)]:
            if self.which != which:
                msg = 'self.which changed while iterating, should be "{}" but is now "{}".'
                raise RuntimeError(msg.format(which, self.which))
            end = min(start+batch_size, N)
            indexes = self.documents_indexes[start:end]
            data = self.dynamic_backend_mosaic(indexes)
            data = data + self.dynamic_cast(rng.normal(0, self.sigma, data.shape))
            ids = list(range(start, end))
            yield data, ids
    
    def batch(self, size=None, index=None, replace=True):
        if index is not None or not replace:
            raise ValueError('not implemented yet')
        if size is None: size = len(self.x.data)
            
        d_ids = self.rng.rng.randint(0, len(self.x.data), (size,2,2))
        d = self.dynamic_backend_mosaic(d_ids)
        d = d + self.rng.normal(0, self.sigma, d.shape)
        
        axis1, axis2 = np.random.randint(0,2,(2, size))
        q_ids = d_ids[range(size), axis1, axis2]
        q = self.x.data[q_ids].reshape(size, 28, 28)
        q = q + self.rng.normal(0, self.sigma, q.shape)
        
        r = (q_ids[:,None,None,None] == d_ids[None]).any(axis=(2,3))
        r = self.dynamic_cast(r)
        return q, d, r