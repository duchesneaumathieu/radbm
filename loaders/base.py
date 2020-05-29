import os
import numpy as np
from radbm.utils.os import StateObj

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

class Loader(StateObj):
    def __init__(self, mode, rng=np.random):
        modes = self.get_available_modes()
        if mode not in modes:
            msg = 'mode must be in {}, got {}'
            raise ValueError(msg.format(modes, mode))
        self.mode=mode
        self.rng=rng
        
    def get_rng_state_hash(self):
        s = self.rng.get_state()
        #don't hash string (e.g. s[0]) since string's hash are salted at each session
        return hash(tuple(s[1])+s[2:])
    
    def __repr__(self):
        s = self.rng.get_state()
        r = 'Loader: {}\nMode: {}\nRNG\'s State (hash): {}'
        return r.format(
            self.__class__.__name__,
            self.mode,
            self.get_rng_state_hash()
        )
    
    def get_available_modes(self):
        return {
            'Uunsupervised',
            'Class',
            'Relational',
            'Relational_list',
            'Relational_matrix',
        }
    
    def get_state(self):
        return self.rng.get_state()
    
    def set_state(self, state):
        #fork rng before updating since it might be global
        self.rng = np.random.RandomState()
        self.rng.set_state(state)
        return self
    
    #for easy sharing rng across multiple object
    def get_rng(self): return self.rng
    def set_rng(self, rng): self.rng=rng; return rng