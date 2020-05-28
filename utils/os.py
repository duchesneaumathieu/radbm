import os, pickle

#not safe yet
def safe_save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    return obj
    
#not safe yet
def safe_load(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj

class StateObj(object):
    def get_state(self):
        raise NotImplementedError()
    
    def set_state(self, state):
        raise NotImplementedError()
    
    def save(self, path=None, file=None):
        state = self.get_state()
        if file is not None: pickle.dump(state, file, -1)
        elif path is not None: safe_save(state, path)
        else: raise ValueError('path or file need to be specified')
        return self
    
    def load(self, path=None, file=None):
        if file is not None: state = pickle.load(file)
        elif path is not None: state = safe_load(path)
        else: raise ValueError('path or file need to be specified')
        self.set_state(state)
        return self