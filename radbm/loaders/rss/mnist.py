import numpy as np
import gzip, pickle, torch
from radbm.loaders.rss import ConjunctiveBooleanRSS
from radbm.utils.fetch import get_directories_list, fetch_file
from radbm.utils.numpy.random import unique_randint, no_subset_unique_randint
from radbm.utils.numpy.function import dihedral4

def mnist_loader(path=None, download=True):
    file_paths = fetch_file('mnist.pkl.gz', path, data_type='dataset', subdirs=['Mnist', 'mnist'], download=download)
    if not file_paths:
        raise FileNotFoundError('could not locate mnist.pkl.gz')
    with gzip.open(file_paths[0], 'rb') as f:
        train_xy, valid_xy, test_xy = pickle.load(f, encoding='latin1')
    return train_xy, valid_xy, test_xy

class MnistCB(ConjunctiveBooleanRSS):
    """
    Conjunctive Boolean RSS with Mnist index terms.

    Parameters
    ----------
    k : int
        The number of index terms per query.
    l : int
        The number of index terms per document.
    m : int
        The total number of index terms.
    n : int
        The database size.
    queries_transform : str (optinal)
        Should be in 'r0', 'r1', 'r2', 'r3', 'sr0', 'sr1', 'sr2' or 'sr3'. It applies a rotation and/or a reflection
        to the index terms of the queries. It follows the Dihedral group syntaxe, e.g. sr2 indicates that a 180 degree
        rotation followed by a vertical reflexion is performed. (default: 'r0' i.e. doing nothing)
    documents_transform : str (optional)
        Same as queries_transform but for the documents' index terms.
    path : str or None (optional)
        The path where to find the Mnist dataset, if None we try to find it in the current directory or in the home directory.
        If it is not found and download is True, it will be downloaded at this location (or in the home directory if it is None).
        (default: None)
    download : bool (optional)
        If we are allowed to download mnist if not found. (default: True)
    mode : str (optional)
        Should be 'balanced' or 'block'. It dictates the behaviour of the batch method (see the batch method). (default: 'balanced')
    which : str (optional)
        Should be 'train', 'valid' or 'test'. The initial dataset to use. Changing this attribute directly will yield
        unknown behaviour. To modify which dataset to use, we need to call train(), valid() or test(). (default : 'train')
    backend : str (optional)
        Should be 'numpy' or 'torch'. It dictates the type of data produced by the batch, iter_queries and iter_documents methods.
        Changing this attribute directly will yield unknown behaviour. To modify which dataset to use, we need to call numpy() or
        torch(). (default : 'numpy')
    device : str (optional)
        Should be 'cpu' or 'cuda' and cannot be 'cuda' if backend is 'numpy'. Similar to backend, to modify it we need to call cpu()
        or cuda(). (default: 'cpu')
    rng : np.random.generator.Generator
        The random number generator used to generate the batches and the database/queries. Should be used for reproducibility.
    """
    def __init__(
        self, k, l, m, n,
        queries_transform='r0', documents_transform='r0',
        path=None, download=True, 
        mode='balanced', which='train',
        backend='numpy', device='cpu',
        rng=np.random):
        
        super().__init__(k, l, m, n, mode=mode, which=which, backend=backend, device=device, rng=rng)
        which_xy = mnist_loader(path, download=download)
        tqx, vqx, wqx = [dihedral4(xy[0].reshape(-1,28,28), queries_transform) for xy in which_xy]
        tdx, vdx, wdx = [dihedral4(xy[0].reshape(-1,28,28), documents_transform) for xy in which_xy]
        self.register_switch('train_qx', tqx)
        self.register_switch('valid_qx', vqx)
        self.register_switch('test_qx', wqx)
        self.register_switch('train_dx', tdx)
        self.register_switch('valid_dx', vdx)
        self.register_switch('test_dx', wdx)
        self.train_group['qx'] = self.train_qx
        self.valid_group['qx'] = self.valid_qx
        self.test_group['qx'] = self.test_qx
        self.train_group['dx'] = self.train_dx
        self.valid_group['dx'] = self.valid_dx
        self.test_group['dx'] = self.test_dx
        getattr(self, self.which)()
    
    def iter_documents(self, batch_size, maximum=np.inf, rng=np.random):
        """
        Generator of the documents in the database with their index.
        
        Parameters
        ----------
        batch_size : int
            The batch size used for each yield.
        maximum : int (optional)
            The maximum number of documents to yield. (default: np.inf)
        
        Yields
        ------
        documents : np.ndarray or torch.Tensor (dtype: float, shape: (batch_size, l, 28, 28))
            A batch of documents.
        indexes : list of int
            The indexes of each documents, i.e. indexes[i] is the index of documents[i].
        """
        for d, i in super().iter_documents(batch_size, maximum=maximum, rng=rng):
            yield self.dx.data[d], i
    
    def iter_queries(self, batch_size, maximum=np.inf, rng=np.random):
        """
        Generator of the queries and their respective relevant documents' index.
        
        Parameters
        ----------
        batch_size : int
            The batch size used for each yield.
        maximum : int (optional)
            The maximum number of documents to yield. (default: np.inf)
        
        Yields
        ------
        queries : np.ndarray or torch.Tensor (dtype: float, shape: (batch_size, k, 28, 28))
            A batch of queries.
        relevants : list of list of int
            The indexes of the relevant documents of each query, i.e. relevants[i] is the relevant documents' 
            list of indexes of for queries[i].
        """
        for q, r in super().iter_queries(batch_size, maximum=maximum, rng=rng):
            yield self.qx.data[q], r
    
    def batch(self, size):
        """
        Parameters
        ----------
        size : int
            The batch size, i.e. the number of queries and documents to return.
            
        Returns
        -------
        queries : np.ndarray or torch.Tensor (dtype: float, shape: (size, k, 28, 28))
            A batch of queries.
        documents : np.ndarray or torch.Tensor (dtype: float, shape: (size, l, 28, 28))
            A batch of documents.
        relevants : np.ndarray or torch.Tensor (dtype: bool, shape: (size,) or (size, size))
            If mode=='balanced', then relevants.shape is (size,) and relevants[i] indicates if queries[i]
            matches with documents[i] (i.e. queries[i]'s index terms is a subset of documents[i]'s index terms).
            The way it is programmed, relevants[:size//2] will always be True while relevants[size//2:] will always
            be False (this is why the mode is called 'balanced').
            Otherwise, if mode=='block', then relevants.shape is (size, size) and relevants[i, j] indicates if queries[i]
            matches with documents[j]. The way it is programmed, the diagonal (relevans[i, i]) is always True while the might
            be True or False dependant on the probability that a random query matches with a random document.
        """
        q, d, r = super().batch(size)
        return self.qx.data[q], self.dx.data[d], r