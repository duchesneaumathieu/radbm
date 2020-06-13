import os, torch
import numpy as np
from radbm.utils.os import StateObj
from radbm.utils.torch import torch_cast, torch_cast_cpu, TorchNumpyRNG

def locate_dataset_dirs(dataset, path):
    # <path> if path is not None
    # $DATASETS_DIR/<dataset>
    # ./datasets/<dataset>
    # ./<dataset>
    paths = list()
    if path is not None:
        if os.path.isfile(path):
            paths.append(os.path.dirname(path))
        else: paths.append(path)
    if 'DATASETS_DIR' in os.environ:
        tmp = set()
        env_path = [p for p in os.environ['DATASETS_DIR'].split(':') if p not in tmp and tmp.add(p) is None and p]
        paths.extend(env_path)
    paths.extend(['./datasets', '.'])
    
    output = list()
    for p in paths:
        while p.endswith('/'):
            p = p[:-1] #remove ending /
        if p.endswith(dataset):
            if os.path.isdir(p):
                output.append(p)
        elif os.path.isdir(os.path.join(p, dataset)):
            output.append(os.path.join(p, dataset))
    return output
    
def locate_dataset_files(dataset, path, file):   
    # <path> if not None (path endswith file)
    # dir/<file> for dir in locate_dataset_dirs
    # <file>
    paths = list()
    if path is not None:
        if path.endswith(file) and os.path.isfile(path):
            paths.append(path)
        else:
            join = os.path.join(path,file)
            if os.path.isfile(join): paths.append(join)
    for d in locate_dataset_dirs(dataset, path):
        f = os.path.join(d,file)
        if os.path.isfile(f): paths.append(f)
    if os.path.isfile(file): paths.append(file)
        
    tmp = set()
    return [p for p in paths if p not in tmp and tmp.add(p) is None and p]
    return paths

class TorchNumpy(object):
    def __init__(self, data, backend, device):
        self.backend = backend
        self.device = device
        if backend not in {'numpy', 'torch'}:
            raise ValueError('backend must be numpy or torch, got {}'.format(backend))
            
        if backend=='numpy' and isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        elif backend=='torch' and isinstance(data, np.ndarray):
            data = torch_cast_cpu(data)
            
        if device=='cpu': self.data = data
        elif device=='cuda':
            if backend=='torch': self.data = data.cuda()
            else: raise ValueError('cannot use cuda with numpy backend')
        else: raise ValueError('device must be cpu or cuda, got {}'.format(device))
        
    def cuda(self):
        self.data = self.data.cuda()
        
    def cpu(self):
        self.data = self.data.cpu()
        
    def torch(self):
        self.data = torch_cast_cpu(self.data)
        
    def numpy(self):
        self.data = self.data.numpy()

class Loader(StateObj):
    def __init__(self, which, backend, device, rng=np.random):
        self.value_check('which', {'train', 'valid', 'test'}, which)
        self.value_check('backend', {'numpy', 'torch'}, backend)
        self.value_check('device', {'cpu', 'cuda'}, device)
        
        if device=='cuda': self.cuda_check()
        if backend=='numpy' and device=='cuda':
            raise ValueError('cannot use cuda with numpy backend')
        
        self.rng=TorchNumpyRNG(rng=rng)
        self.switch_attrs = ['rng']
        self.which=which
        self.backend='numpy'
        self.device='cpu'
        if backend=='torch': self.torch()
        if device=='cuda': self.cuda()
            
        self.train_group = dict()
        self.valid_group = dict()
        self.test_group = dict()
        getattr(self, which)()
    
    def value_check(self, name, valids, value):
        if value not in valids:
            msg = '{} must be in {}, got {}'
            raise ValueError(msg.format(name, valids, value))
            
    def cuda_check(self):
        if not torch.cuda.is_available():
            raise ValueError('cannot use cuda, it is not available')
    
    def register_switch(self, name, data):
        if hasattr(self, name):
            raise ValueError('{} already used'.format(name))
        tnp = TorchNumpy(data, self.backend, self.device)
        setattr(self, name, tnp)
        self.switch_attrs.append(name)
    
    def apply_switch(self, function_name):
        for name in self.switch_attrs:
            getattr(getattr(self, name), function_name)()
    
    def torch(self):
        if self.backend == 'torch': return self
        self.apply_switch('torch')
        self.backend = 'torch'
        return self
        
    def numpy(self):
        if self.backend == 'numpy': return self
        if self.device == 'cuda':
            msg = 'cannot use numpy backend with cuda, try loader.cpu() first'
            raise ValueError(msg)
        self.apply_switch('numpy')
        self.backend = 'numpy'
        return self
        
    def cuda(self):
        if self.device == 'cuda': return self
        if self.backend == 'numpy':
            msg = 'cannot use cuda with numpy backend, try loader.torch() first'
            raise ValueError(msg)
        self.cuda_check()
        self.apply_switch('cuda')
        self.device = 'cuda'
        return self
        
    def cpu(self):
        if self.device == 'cpu': return self
        self.apply_switch('cpu')
        self.device = 'cpu'
        return self
    
    def unpack_group(self, group):
        for k, v in getattr(self, '{}_group'.format(self.which)).items():
            setattr(self, k, None)
            
        for k, v in group.items():
            if not hasattr(self, k) or getattr(self, k) is None:
                setattr(self, k, v)
    
    def train(self):
        self.unpack_group(self.train_group)
        self.which='train'
        return self
        
    def valid(self):
        self.unpack_group(self.valid_group)
        self.which='valid'
        return self
        
    def test(self):
        self.unpack_group(self.test_group)
        self.which='test'
        return self
     
    def dynamic_cast(self, data):
        if self.backend=='torch' and isinstance(data, np.ndarray):
            data = torch_cast_cpu(data)
        if self.backend=='numpy' and isinstance(data, torch.Tensor):
            data = torch.cpu().numpy()
        if isinstance(data, torch.Tensor):
            if self.device=='cuda' and data.device.type=='cpu':
                data = data.cuda()
            if self.device=='cpu' and data.device.type=='cuda':
                data = data.cpu()
        return data
            
    def __repr__(self):
        s = self.rng.get_state()
        r = 'Loader: {} [{} ({})]\nSet: {}\nRNG\'s State (hash): {}'
        return r.format(
            self.__class__.__name__,
            self.backend,
            self.device,
            self.which,
            self.rng.get_state_hash()
        )
    
    def get_state(self):
        return self.rng.get_state()
    
    def set_state(self, state):
        #fork rng before updating since it might be use elsewhere (e.g. the global np.random)
        #for using the same rng across multiple objects, use set_rng
        self.rng = np.random.RandomState()
        self.rng.set_state(state)
        return self
    
    #for easy sharing rng across multiple object
    def get_rng(self):
        return self.rng.rng
    
    def set_rng(self, rng):
        if isinstance(rng, TorchNumpyRNG): self.rng = rng
        else: self.rng=TorchNumpyRNG(rng)
        return self
    
class IRLoader(Loader):
    def __init__(self, mode, which, backend, device, rng=np.random):
        super().__init__(which, backend, device, rng=rng)
        self.value_check('mode', self.get_available_modes(), mode)
        self.mode = mode
        
    def __repr__(self):
        s = self.rng.get_state()
        r = 'Loader: {} [{} ({})]\nSet: {}\nBatch Mode: {}\nRNG\'s State (hash): {}'
        return r.format(
            self.__class__.__name__,
            self.backend,
            self.device,
            self.which,
            self.mode,
            self.rng.get_state_hash()
        )
    
    def get_available_modes(self):
        return {
            'unsupervised',
            'class_relation',
            'relational_triplets',
            'relational_matrix'
        }