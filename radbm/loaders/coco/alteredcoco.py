import h5py, torch
import numpy as np
from radbm.loaders.base import IRLoader
from radbm.utils.fetch import fetch_file
from .alterimages import AlterImages

def coco_finder(which, path=None):
    file_paths = fetch_file(
        which,
        path,
        data_type='dataset',
        subdirs=['MSCoco', 'Coco'],
        download=False
    )
    if not file_paths:
        raise FileNotFoundError('could not locate squared_coco.hdf5')
    return h5py.File(file_paths[0], 'r')

class AlteredCoco(IRLoader):
    def __init__(
        self, batch_global_scale_range=[1.,1.], generator=torch.default_generator, path=None,
        mode='relational_matrix', which='train', backend='numpy', device='cpu', rng=np.random):
        super().__init__(mode=mode, which=which, backend=backend, device=device, rng=rng)
        if mode is not 'relational_matrix':
            raise NotImplementedError('mode={} not implemented yet'.format(mode))
        self.original_coco = coco_finder('squared_coco.hdf5', path)
        self.altered_coco = coco_finder('altered_coco.hdf5', path)
        self.alter = AlterImages((256, 256), batch_global_scale_range=batch_global_scale_range, generator=generator)
        N = len(self.original_coco['data'])
        self.train_group = {
            'original_indexes':np.arange(60000,N-40000),
            'altered_indexes': np.arange(0,10000)
        }
        self.valid_group = {
            'original_indexes':np.array(list(range(0,60000)) + list(range(N-40000,N))),
            'altered_indexes': np.arange(0,10000)
        }
        getattr(self, self.which)()
    
    def close(self):
        self.original_coco.close()
        self.altered_coco.close()
        
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
    
    def get_available_modes(self):
        return {
            #'relational_pairs', #(x,k,r)
            #'relational_triplets', #(x, pos_x, neg_x)
            'relational_matrix', #(xs, ks, R)
        }
    
    def get_relation_prob(self):
        return 1/100000
    
    def get_relation_log_prob(self):
        return -np.log(100000)
    
    def iter_queries(self, batch_size, maximum=np.inf):
        which = self.which
        coco_id = '{}_coco_id'.format(which)
        N = min(maximum, len(self.altered_indexes))
        nbatch = int(np.ceil(N//batch_size))
        for start in [i*batch_size for i in range(nbatch)]:
            if self.which != which:
                msg = 'self.which changed while iterating, should be "{}" but is now "{}".'
                raise RuntimeError(msg.format(which, self.which))
            end = min(start+batch_size, N)
            indexes = list(self.altered_indexes[start:end])
            data = self.altered_coco[which][indexes]/255
            relevant = self.altered_coco[coco_id][indexes]
            yield self.dynamic_cast(data), [{int(r)} for r in relevant]
    
    def iter_documents(self, batch_size, maximum=np.inf):
        which = self.which
        N = min(maximum, len(self.original_indexes))
        nbatch = int(np.ceil(N//batch_size))
        for start in [i*batch_size for i in range(nbatch)]:
            if self.which != which:
                msg = 'self.which changed while iterating, should be "{}" but is now "{}".'
                raise RuntimeError(msg.format(which, self.which))
            end = min(start+batch_size, N)
            indexes = list(self.original_indexes[start:end])
            data = self.original_coco['data'][indexes]/255
            ids = self.original_coco['coco_id'][indexes]
            yield self.dynamic_cast(data), list(ids)
            
    def choice_without_replacement(self, N, size):
        #faster than np.choice for size << N
        for _ in range(5): #not using while True because it is better late than never 
            ids = self.rng.rng.randint(0, N, size)
            if len(ids)==len(set(ids)):
                return ids
        return self.rng.rng.choice(range(N), size=size, replace=False)
    
    def batch(self, size=None, index=None, replace=True):
        if index is not None or not replace:
            raise ValueError('not implemented yet')
        ids = self.choice_without_replacement(len(self.original_indexes), size)
        indexes = sorted(self.original_indexes[ids])
        d = self.original_coco['data'][indexes]/255
        d = torch.tensor(d, dtype=torch.float32, device=self.alter.device)
        q = self.alter(d)
        r = self.dynamic_cast(np.eye(size, dtype=np.bool))
        return self.dynamic_cast(q), self.dynamic_cast(d), r