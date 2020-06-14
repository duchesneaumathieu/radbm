import os, pickle

#not safe yet
def safe_save(obj, path):
    """
    Safely saves obj in path by creating .tmp file
    before ovewriting an existing file

    Parameters
    ----------
    obj : object
        the object to be saved
    path : str
        the path where to save obj
    
    Returns
    -------
    obj : object
        the same obj received as input

    """
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    return obj
    
#not safe yet
def safe_load(path):
    """
    Safely load an object if .tmp is present 
    it will be prefered unless it is corrupted
    (i.e. pickle.load fails)

    Parameters
    ----------
    path : str
        the path where to object is saved
    
    Returns
    -------
    obj : object
        the object loaded

    """
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
