from .hamming import (
    hamming_distance,
    conditional_distance_counts,
    conditional_hamming_counts,
    hamming_pr_curve,
)
from .subset import subset_distance

# This lets Sphinx know you want to document binary.hamming_multi_probing.HammingMultiProbing as binary.HammingMultiProbing.
# Solution from https://stackoverflow.com/a/66996523
__all_exports = [
    hamming_distance,
    conditional_distance_counts,
    conditional_hamming_counts,
    hamming_pr_curve,
    subset_distance,
]

for e in __all_exports:
    e.__module__ = __name__

__all__ = [e.__name__ for e in __all_exports]