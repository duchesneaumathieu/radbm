from .basic import DictionarySearch, KeyValueHeap
from .binary import (
    HammingMultiProbing, #not in __all__ but still accessible (from radbm.search import HammingMultiProbing)
    BernoulliMultiProbing,
)
from .reduction import (
    HammingReduction, #not in __all__ but still accessible
    BernoulliReduction, #not in __all__ but still accessible
)

# This lets Sphinx know you want to document binary.hamming_multi_probing.HammingMultiProbing as binary.HammingMultiProbing.
# Solution from https://stackoverflow.com/a/66996523
__all_exports = [
    DictionarySearch,
    KeyValueHeap,
]

for e in __all_exports:
    e.__module__ = __name__

__all__ = [e.__name__ for e in __all_exports]