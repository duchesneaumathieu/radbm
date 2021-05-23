import os, sys, time, torch, random

def _get_tmp_path(path):
    return os.path.join(os.path.dirname(path), '.tmp_' + os.path.basename(path))
    
def _try_load(path, map_location, n_attempts, stderr):
    #try loading path
    for attempt in range(n_attempts):
        try:
            with open(path, 'rb') as f:
                time.sleep(random.uniform(attempt*.1, attempt*.5))
                obj = torch.load(f, map_location=map_location)
            return obj
        except Exception as e:
            last_err = e
            msg = 'loading {} failed on attempt {}/{} with error: {}\n'
            stderr.write(msg.format(path, attempt+1, n_attempts, repr(e)))
            stderr.flush()
            continue
        break
    raise last_err

def safe_save(
    obj, path,
    n_attempts = 5,
    n_save_attempts = 2,
    n_reload_attempts = 2,
    verbose = False,
    stdout = sys.stdout,
    stderr = sys.stderr,
    pickle_protocol=torch.serialization.DEFAULT_PROTOCOL
):
    """
    Safely saves obj in path by creating .tmp_<name> file
    before ovewriting an existing file then trying to read the file
    before removing .tmp_<name>

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
    tmp_path = _get_tmp_path(path)
    path_exists = os.path.isfile(path)
    tmp_path_exists = os.path.isfile(tmp_path)
    if path_exists and not tmp_path_exists:
        if verbose:
            stdout.write('appending .tmp to {}.\n'.format(path))
            stdout.flush()
        os.rename(path, tmp_path)
    elif path_exists and tmp_path_exists:
        if verbose:
            stdout.write('existing path and tmp, testing if path is corrupted.\n')
            stdout.flush()
        try:
            _try_load(path, map_location=None, n_attempts=5, stderr=stderr)
            corrupted = False
        except:
            corrupted = True
        if corrupted:
            if verbose:
                stdout.write('path is corrupted, trying to overwrite path (keeping tmp).\n')
                stdout.flush()
        else:
            if verbose:
                stdout.write('path is not corrupted, overwriting tmp.\n')
                stdout.flush()
            os.rename(path, tmp_path)
        
    success = False
    for attempt in range(n_attempts):
        
        #try saving
        save_success = False
        for save_attempt in range(n_save_attempts):
            time.sleep(random.uniform(save_attempt*.1, save_attempt*.5)) #delay
            try:
                with open(path, 'wb') as f:
                    torch.save(obj, f, pickle_protocol=pickle_protocol)
            except Exception as e:
                last_err = e
                msg = 'saving {} failed on attempt {}/{} with error: {}\n'
                stderr.write(msg.format(path, save_attempt+1, n_save_attempts, repr(e)))
                stderr.flush()
                continue
            save_success = True
            break
            
        #try reloading
        reload_success = False
        if save_success:
            for reload_attempt in range(n_reload_attempts):
                time.sleep(random.uniform(reload_attempt*.1, reload_attempt*.5)) #delay
                try: #load
                    with open(path, 'rb') as f:
                        reloaded_obj = torch.load(f)
                except Exception as e:
                    last_err = e
                    msg = 'reloading {} failed on attempt {}/{} with error: {}\n'
                    stderr.write(msg.format(path, reload_attempt+1, n_reload_attempts, repr(e)))
                    stderr.flush()
                    #delay
                    continue
                reload_success = True
                break
        
        success = save_success and reload_success
        if success: break
        else:
            msg = 'saving + reloading failed on attempt {}/{} for {}\n'
            stderr.write(msg.format(attempt+1, n_attempts, path))
            stderr.flush()
        
    if not success:
        if verbose:
            stdout.write('saving failed. Raising last error.\n')
            stdout.flush()
        if os.path.isfile(tmp_path):
            if verbose:
                stdout.write('renaming tmp to path.\n')
                stdout.flush()
            os.rename(tmp_path, path)
        raise last_err #raise last error
        
    if verbose:
        stdout.write('saving succeeded.\n')
        stdout.flush()
    if os.path.isfile(tmp_path):
        if verbose:
            stdout.write('removing tmp.\n')
            stdout.flush()
        os.remove(tmp_path)
        
    return obj
    
def safe_load(path, map_location=None, n_attempts=5, verbose=False, stdout=sys.stdout, stderr=sys.stderr):
    """
    Safely load an object using the possibly existing path/to/file/.tmp_<name>

    Parameters
    ----------
    path : str
        the path where the object is saved
    
    Returns
    -------
    obj : object
        the object loaded

    """
    tmp_path = _get_tmp_path(path)
    path_exists = os.path.isfile(path)
    tmp_path_exists = os.path.isfile(tmp_path)
    if path_exists and not tmp_path_exists: #usual
        if verbose:
            stdout.write('trying to load path.\n')
            stdout.flush()
        return _try_load(path, map_location, n_attempts=n_attempts, stderr=stderr)
    elif not path_exists and tmp_path_exists:
        #unlikely, it implies safe_save did not finish.
        if verbose:
            stdout.write('path not found but tmp exists. Renaming tmp to path and trying to load.\n')
            stdout.flush()
        os.rename(tmp_path, path) #cleaning
        return _try_load(path, map_location, n_attempts=n_attempts, stderr=stderr)
    elif path_exists and tmp_path_exists:
        #unlikely, it implies safe_save did not finish.
        if verbose:
            stdout.write('path and tmp exist. trying to load path.\n')
            stdout.flush()
        path_loaded = False
        try: 
            obj = _try_load(path, map_location, n_attempts=n_attempts, stderr=stderr)
            path_loaded = True
        except: pass #path is corrupted
        if path_loaded:
            if verbose:
                stdout.write('path loaded, removing tmp.\n')
                stdout.flush()
            os.remove(tmp_path)
            return obj
        if verbose:
            stdout.write('failed to load path, trying to load tmp.\n')
            stdout.flush()
        return _try_load(tmp_path, map_location, n_attempts=n_attempts, stderr=stderr)
    else:
        msg = '{} or its .tmp file do not exist.'
        raise FileNotFoundError(msg.format(path))

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
        try:
            state = safe_load(path, *args, **kwargs)
            self.set_state(state)
        except FileNotFoundError as e:
            if if_inexistent=='raise':
                msg = '{} does not exists and if_inexistent is "raise"'
                raise FileNotFoundError(msg.format(path)) from e
        return self