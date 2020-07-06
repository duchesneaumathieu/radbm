import os, json, gdown
from pathlib import Path

_path = os.path.dirname(__file__)
_fileid_path = os.path.join(_path, 'fileid.json')
with open(_fileid_path, 'r') as f:
    _fileid = json.load(f)

    
def available_files():
    """
    Returns
    -------
    files : list of str
        The files available for download 
    """
    return list(_fileid.keys())
    
def download_file(file, path=None, verbose=False):
    """
    Download a file from Google Drive. This uses the gdown package.
    Warning, this function overwrite any existing file. 
    
    Parameters
    ----------
    file : str
        The name of the file to download. Use available_files() to 
        see which files are available for download.
    path : str, optional (default=file)
        The path where to download the file. If the path does not exists
        it will be created. This should contain the filename.
    verbose : bool, optional (default=False)
        If True output the download progress.
        
    Returns
    -------
    path : str
        The path where the file has been downloaded.
    """
    path = file if path is None else path
    directory = os.path.dirname(path)
    Path(directory).mkdir(parents=True, exist_ok=True)
    return gdown.download(_fileid[file], path, quiet=not verbose)