import os, torch
import numpy as np
from radbm.utils.os import StateObj
from radbm.utils.torch import (
    TorchNumpy,
    TorchNumpyRNG,
    torch_cast_cpu
)

class Loader(StateObj):
    """
    An abstract class managing numpy vs torch, cpu vs gpu and train
    vs valid vs test. This should be subclassed with a particular
    dataset (i.e. Mnist).
    
    Parameters
    ----------
    which : str
        Which datasets version to use. Should be 'train', 'valid' or 'test'.
    backend : str
        Which backend to use, should be 'numpy' or 'torch'.
    device : str
        Which device to use, should be 'cpu' or 'cuda'. 'cuda' is only
        available if backend=='torch'.
    rng : numpy.random.RandomState
        A random number generator for reproducibility.
    """
    def __init__(self, which, backend, device, rng=None):
        rng = np.random if rng is None else rng
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
        self.end_init()
        
    def end_init(self):
        getattr(self, self.which)()
    
    def value_check(self, name, valids, value):
        if value not in valids:
            msg = '{} must be in {}, got {}'
            raise ValueError(msg.format(name, valids, value))
            
    def cuda_check(self):
        if not torch.cuda.is_available():
            raise ValueError('cannot use cuda, it is not available')
    
    def register_switch(self, name, data):
        """
        This function should only be used when subclassing. This is to
        register data for when a user will call: numpy(), torch(), cpu()
        or cuda(). Each value will be transfered to the appropriate format.
        
        Parameters
        ----------
        name : str
            The name of the data. setarrt is used so one could later do self.<name>
            to reach the data.
        data : numpy.ndarray or torch.Tensor
            The data to register.
        """
        if hasattr(self, name):
            raise ValueError('{} already used'.format(name))
        tnp = TorchNumpy(data, self.backend, self.device)
        setattr(self, name, tnp)
        self.switch_attrs.append(name)
    
    def apply_switch(self, function_name):
        for name in self.switch_attrs:
            getattr(getattr(self, name), function_name)()
    
    def torch(self):
        """
        Converts each registered data (using register_switch) into torch.Tensor
        """
        if self.backend == 'torch': return self
        self.apply_switch('torch')
        self.backend = 'torch'
        return self
        
    def numpy(self):
        """
        Converts each registered data (using register_switch) into numpy.ndarray
        
        Raises
        ------
        ValueError
            If device=='cuda'
        """
        if self.backend == 'numpy': return self
        if self.device == 'cuda':
            msg = 'cannot use numpy backend with cuda, try loader.cpu() first'
            raise ValueError(msg)
        self.apply_switch('numpy')
        self.backend = 'numpy'
        return self
        
    def cuda(self):
        """
        Transfers each registered data (using register_switch) to the GPU
        
        Raises
        ------
        ValueError
            If backend=='numpy'
        """
        if self.device == 'cuda': return self
        if self.backend == 'numpy':
            msg = 'cannot use cuda with numpy backend, try loader.torch() first'
            raise ValueError(msg)
        self.cuda_check()
        self.apply_switch('cuda')
        self.device = 'cuda'
        return self
        
    def cpu(self):
        """
        Transfers each registered data (using register_switch) to the CPU
        """
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
        """
        Switch to training dataset.
        """
        self.unpack_group(self.train_group)
        self.which='train'
        return self
        
    def valid(self):
        """
        Switch to validation dataset.
        """
        self.unpack_group(self.valid_group)
        self.which='valid'
        return self
        
    def test(self):
        """
        Switch to testing dataset.
        """
        self.unpack_group(self.test_group)
        self.which='test'
        return self
     
    def dynamic_cast(self, data):
        """
        Cast data according to the current state of the class.
        E.g. when backend=='torch', device=='cuda' and the inputed
        data is numpy.ndarray, the array will be converted to
        torch.Tensor and transfered on the GPU.
        
        Parameters
        ----------
        data : numpy.ndarray or torch.Tensor
            The data to cast
            
        Returns
        -------
        casted_data : numpy.ndarray or torch.Tensor
            The casted data
        """
        if self.backend=='torch' and isinstance(data, np.ndarray):
            data = torch_cast_cpu(data)
        if self.backend=='numpy' and isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
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
        rng = np.random.RandomState()
        rng.set_state(state)
        self.set_rng(rng)
        return self
    
    #for easy sharing rng across multiple object
    def get_rng(self):
        """
        Utility method to get the rng.
        
        Returns
        -------
        rng : numpy.random.RandomState
            The rng used inside the utility class TorchNumpyRNG.
        """
        return self.rng.rng
    
    def set_rng(self, rng):
        """
        Utility method to set the rng.
        
        Parameters
        ----------
        rng : numpy.random.RandomState or TorchNumpyRNG
            The rng to use going forward.
        
        Returns
        -------
        self : Loader
        """
        if isinstance(rng, TorchNumpyRNG):
            #rewrap for consistency
            self.rng = TorchNumpyRNG(rng.rng)
        else:
            self.rng = TorchNumpyRNG(rng)
        
        #casting the numpy-cpu rng if necessary
        if self.backend=='torch':
            self.rng.torch()
        if self.device=='cuda':
            self.rng.cuda()
        return self
    
class IRLoader(Loader):
    """
    A subclass of Loader meant for Information Retrieval. This introduces the
    notion of mode which gouverns to way batches will be given.
    
    Parameters
    ----------
    mode : str
        should be in IRLoader.get_available_modes()
    """
    def __init__(self, mode, which, backend, device, rng=None):
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