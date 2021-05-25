from .pre_ap import (
    pre_average_precision,
    pre_mean_average_precision,
    batch_pre_average_precision,
)

# This lets Sphinx know you want to document binary.hamming_multi_probing.HammingMultiProbing as binary.HammingMultiProbing.
# Solution from https://stackoverflow.com/a/66996523
__all_exports = [
    pre_average_precision,
    pre_mean_average_precision,
    batch_pre_average_precision,
]

for e in __all_exports:
    e.__module__ = __name__

__all__ = [e.__name__ for e in __all_exports]