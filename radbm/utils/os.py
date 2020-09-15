import os, torch

#not safe yet
def safe_save(obj, path, pickle_protocol=torch.serialization.DEFAULT_PROTOCOL):
    """
    Safely saves obj in path by creating .tmp file
    before ovewriting an existing file

    Parameters
    ----------
    obj : object
        the object to be saved
    path : str
        the path where to save obj
    pickle_protocol : int (optional)
        The pickle protocol to uses. By default, the torch default_protocol
        is used.
    
    Returns
    -------
    obj : object
        the same obj received as input

    """
    with open(path, 'wb') as f:
        torch.save(obj, f, pickle_protocol=pickle_protocol)
    return obj
    
#not safe yet
def safe_load(path, map_location=None):
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
        obj = torch.load(f, map_location=map_location)
    return obj

class StateObj(object):
    def get_state(self):
        raise NotImplementedError()
    
    def set_state(self, state):
        raise NotImplementedError()
    
    def save(self, path, if_exists='overwrite', *args, **kwargs):
        #TODO replace *args **kwargs with the right names
        if not os.path.exists(path) or if_exists=='overwrite':
            state = self.get_state()
            safe_save(state, path, *args, **kwargs)
        elif if_exists=='raise':
            msg = '{} exists and if_exists is "raise"'
            raise FileExistsError(msg.format(path))
        return self
    
    def load(self, path, if_inexistent='raise', *args, **kwargs):
        #TODO replace *args **kwargs with the right names
        if os.path.exists(path):
            state = safe_load(path, *args, **kwargs)
            self.set_state(state)
        elif if_inexistent=='raise':
            msg = '{} does not exists and if_inexistent is "raise"'
            raise FileNotFoundError(msg.format(path))
        return self
