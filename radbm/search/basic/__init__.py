from .dictionary import DictionarySearch
from .heap import KeyValueHeap
from .trie import Trie

__all_exports = [
    DictionarySearch,
    KeyValueHeap,
    Trie,
]

for e in __all_exports:
    e.__module__ = __name__

__all__ = [e.__name__ for e in __all_exports]