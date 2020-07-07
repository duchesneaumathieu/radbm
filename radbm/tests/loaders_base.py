import torch, unittest
import numpy as np
from radbm.loaders.base import Loader, IRLoader

class DummyLoader(Loader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #registering each dataset to switch
        #this is only possible if the datasets are small
        self.register_switch('train_dataset', 1*np.ones((2,3)))
        self.register_switch('valid_dataset', 2*np.ones((2,3)))
        self.register_switch('test_dataset', 3*np.ones((2,3)))
        
        #updating <which> groups.
        #that way, e.g. when self.valid() will be called
        #self.dataset will contain self.valid_dataset
        self.train_group['dataset'] = self.train_dataset
        self.valid_group['dataset'] = self.valid_dataset
        self.test_group['dataset'] = self.test_dataset
        self.end_init()
    
class TestLoader(unittest.TestCase):
    def setUp(self):
        self.dummy = DummyLoader('train', 'numpy', 'cpu')
        
    def try_switch_cuda(self):
        if torch.cuda.is_available():
            self.dummy.cuda()
            self.assertCUDA()
        else:
            with self.assertRaises(ValueError):
                self.dummy.cuda()
        
    def test_init(self):
        #test value_check
        with self.assertRaises(ValueError):
            self.dummy = DummyLoader('training', 'numpy', 'cpu')
        
        self.dummy = DummyLoader('train', 'numpy', 'cpu')
        self.assertDatasetValue(1)
        self.assertNumpy()
        
        self.dummy = DummyLoader('valid', 'numpy', 'cpu')
        self.assertDatasetValue(2)
        self.assertNumpy()
        
        #test numpy + cuda init error
        with self.assertRaises(ValueError):
            self.dummy = DummyLoader('valid', 'numpy', 'cuda')
        
        self.dummy = DummyLoader('valid', 'torch', 'cpu')
        self.assertDatasetValue(2)
        self.assertTorch()
        self.assertCPU()
        
        if torch.cuda.is_available():
            self.dummy = DummyLoader('test', 'torch', 'cuda')
            self.assertDatasetValue(3)
            self.assertTorch()
            self.assertCUDA()
        else:
            with self.assertRaises(ValueError):
                self.dummy = DummyLoader('test', 'torch', 'cuda')

        self.dummy.cpu().numpy()
        
        #test register_switch 'already exists' error
        with self.assertRaises(ValueError):
            self.dummy.register_switch('dataset', np.array(0))
        
    def assertNumpy(self):
        self.assertTrue(isinstance(self.dummy.dataset.data, np.ndarray))
        
    def assertTorch(self):
        self.assertTrue(isinstance(self.dummy.dataset.data, torch.Tensor))
        
    def assertCPU(self):
        self.assertTrue(self.dummy.dataset.data.device.type=='cpu')
        
    def assertCUDA(self):
        self.assertTrue(self.dummy.dataset.data.device.type=='cuda')
        
    def test_backend_and_device(self):
        #test init backend
        self.assertNumpy()
        
        #test cuda switch error
        with self.assertRaises(ValueError):
            self.dummy.cuda()
        
        #test switch to torch
        self.dummy.torch()
        self.assertTorch()
        self.assertCPU()
        
        #test switch to CUDA
        if torch.cuda.is_available():
            self.dummy.cuda()
            self.assertCUDA()

            #test numpy switch error
            with self.assertRaises(ValueError):
                self.dummy.numpy()
        else:
            with self.assertRaises(ValueError):
                self.dummy.cuda()
        
        #test switch to CPU
        self.dummy.cpu()
        self.assertCPU()
        
        #test switch to (back) numpy
        self.dummy.numpy()
        self.assertNumpy()
        
    def assertDatasetValue(self, value):
        eq = self.dummy.dataset.data==value
        self.assertTrue(eq.all())
    
    def _test_which(self):
        self.dummy.train()
        self.assertDatasetValue(1)
        
        self.dummy.valid()
        self.assertDatasetValue(2)
        
        self.dummy.test()
        self.assertDatasetValue(3)
    
    def test_which(self):
        self._test_which()
        
        self.dummy.torch()
        self._test_which()
        
        self.try_switch_cuda()
        self._test_which()
        
        self.dummy.cpu().numpy()
        self._test_which()
        
    def test_dynamic_cast(self):
        data = torch.tensor(0, device='cpu')
        
        #torch to numpy
        data = self.dummy.dynamic_cast(data)
        self.assertTrue(isinstance(data, np.ndarray))
        
        self.dummy.torch()
        data = self.dummy.dynamic_cast(data)
        self.assertTrue(isinstance(data, torch.Tensor))
        self.assertTrue(data.device.type=='cpu')
        
        if torch.cuda.is_available():
            self.dummy.cuda()
            data = self.dummy.dynamic_cast(data)
            self.assertTrue(isinstance(data, torch.Tensor))
            self.assertTrue(data.device.type=='cuda')
        
        self.dummy.cpu()
        data = self.dummy.dynamic_cast(data)
        self.assertTrue(isinstance(data, torch.Tensor))
        self.assertTrue(data.device.type=='cpu')
        
        self.dummy.numpy()
        
    def test_utilities(self):
        repr(self.dummy) #make sure it does not crash
        
        state = self.dummy.get_state()
        self.dummy.set_state(state)
        
        rng = self.dummy.get_rng()
        self.dummy.set_rng(rng) #numpy.random.RandomState
        self.dummy.set_rng(self.dummy.rng) #TorchNumpyRNG
        
    def test_without_cuda(self):
        is_available = torch.cuda.is_available
        torch.cuda.is_available = lambda : False
        try:
            #retest everything without cuda
            self.test_init()
            self.test_backend_and_device()
            self.test_which()
            self.test_dynamic_cast()
            self.test_utilities()
        finally:
            torch.cuda.is_available = is_available
            
class TestIRLoader(unittest.TestCase):
    def test_utilities(self):
        modes = IRLoader.get_available_modes(None)
        mode = next(iter(modes))
        dummy = IRLoader(mode, 'train', 'numpy', 'cpu')
        repr(dummy)