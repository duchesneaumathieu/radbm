## Utility for loading datasets for training and evaluations

### Module's Mision
 * Intuitive
 * Repeatability (seeded)
 * Save/load loader states (if crash or interuption occurs)
 * Standardized usage
 
### Usage
```python
   from radbm.loaders.mnist import NoisyMnist
   
   loader = NoisyMnist(sigma=0.2, exp_name='my_exp', mode='relational', rng=np.random)
   x, d, r = loader.batch(which='train', batch_size=32)
   r_hat = f(x, d)
   l = loss(r, r_hat)
   loader.save()
```

### Modes
 * unsupervised (x,)
 * class (x, c)
 * relational (x, d, r)
 * relationals_list (X, D, R)
 * relationals_matrix (X, D, R)

### Methods
 * dump(self, path=None, file=None) -> state
 * dumps(self) -> state
 * load(self, path=None, file=None) -> self
 * loads(self, state) -> self
 * build_queries(self, which='train', size=None, index=None, replace=True) -> X
 * build_documents(self, which='train', size=None, index=None, replace=True)
    * -> D if relation or relational_matrix or relation_list
    * -> Error otherwise
 * batch(self, which='train', size=None, index=None, replace=True)
    * -> X if mode=unsupervised
    * -> X, C if mode=class
    * -> X^n, D^n, R^n if mode=relational
    * -> X^n, D^m. R^(nxm) if mode=relational_matrix
 * get_epoch(self, batch_size, rng=np.random) -> Epoch
