import os
from radbm.utils import unique_list
from radbm.utils.gdrive import download_file

DATASETS_DIR = 'DATASETS_DIR'
MODELS_DIR = 'MODELS_DIR'
PACKAGE_VAR = 'PYTHONRADBM_DUMP'

def get_directories_list(path=None, data_type=None):
    """
    This function return the list of directory that is relevant for
    RADBM in order, it returns:

    path
    $DATASETS_DIR if data_type=='dataset'
    $MODELS_DIR if data_type=='model'
    $PYTHONRADBM_DUMP/<data_type>
    $HOME/.radbm/<data_type>
    .
    
    Parameters
    ----------
    path : str, optional
        The path to look first, this is helpful when a user specifies a
        path to file or relevant directory
    data_type : str, optional
        should be 'dataset' or 'model' this tell to lookup for
        environment variables specific to dataset or model
        
    Returns
    -------
    dirs : list of str
        The relevant directories in order
    """
    dirs = list()
    
    #path directory
    if path is not None:
        dirs.append(os.path.abspath(path))
    
    #data_type directories
    if data_type=='dataset':
        if DATASETS_DIR in os.environ:
            paths = os.environ[DATASETS_DIR].split(':')
            dirs.extend(unique_list(paths))
    elif data_type=='model':
        if MODELS_DIR in os.environ:
            paths = os.environ[MODELS_DIR].split(':')
            dirs.extend(unique_list(paths))
    elif data_type is not None:
        msg = 'data_type must be dataset or model, got {}'
        raise ValueError(msg.format(data_type))
    
    #package environment variable directory
    if PACKAGE_VAR in os.environ:
        path = os.environ[PACKAGE_VAR]
        if data_type is not None:
            path = os.path.join(path, data_type+'s')
        dirs.append(path)
        
    #package default direcotry
    if 'HOME' in os.environ:
        path = os.path.join(os.environ['HOME'], '.radbm')
        if data_type is not None:
            path = os.path.join(path, data_type+'s')
        dirs.append(path)
        
    #adding current direcory
    dirs.append(os.getcwd())
    
    return dirs

def expend_paths(paths, subpaths):
    """
    Parameters
    ----------
    dirs : list of str (directory path)
    subpaths : list of str (sub-directory path)
    """
    expended_paths = list()
    for path in paths:
        for subpath in subpaths:
            fullpath = os.path.join(path, subpath)
            expended_paths.append(fullpath)
    return expended_paths
    
def fetch_file(file, path=None, data_type=None, subdirs=None, download=True):
    """
    lookup on the machine for file otherwise download it.
    
    Parameters
    ----------
    file : str
        The name of the file to fetch
    path : str, optional (default=None)
        The principal path to look for the file. If the find is not
        found, this function will attemp to download to path to this file.
    data_type : str ('dataset' or 'model'), optional (default=None)
        Modifies the path to consider when looking for the file. If path is
        None and the file is not found, it will affect where to the file is
        downloaded, see get_directories_list for more information.
    subdirs : list of str, optional (default=None)
       Additional sub-directories to lookup
    download : bool
        A boolean to indicate if we want to download the file if not found
        on the machine
        
    Returns
    -------
    paths_list : list of str
        The list of path where to find the file
    """
    #Lookup
    dirs_list = get_directories_list(path, data_type=data_type)
    if subdirs is not None:
        dirs_list = expend_paths(dirs_list, subdirs)
    
    paths_list = expend_paths(dirs_list, [file])
    exists_paths_list = [path for path in paths_list if os.path.isfile(path)]
    
    if exists_paths_list:
        #file found
        return exists_paths_list
    elif download:
        #file not found -> download
        download_path = paths_list[0] #not empty, at least './<file>' in paths_list
        try:
            download_file(file, download_path)
        except KeyError as e:
            msg = '{} not found and is not registered for download'
            raise FileNotFoundError(msg.format(file)) from e
        return [download_path]
    else:
        msg = '{} not found and download disabled'
        raise FileNotFoundError(msg.format(file))