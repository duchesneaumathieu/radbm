from .hamming_multi_probing import HammingMultiProbing

# This lets Sphinx know you want to document binary.hamming_multi_probing.HammingMultiProbing as binary.HammingMultiProbing.
# Solution from https://stackoverflow.com/a/66996523
__all_exports = [HammingMultiProbing]

for e in __all_exports:
    e.__module__ = __name__

__all__ = [e.__name__ for e in __all_exports]