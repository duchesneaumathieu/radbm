from .sorted_merge import sorted_merge
from .smallest_subset_sums import smallest_subset_sums
from .likeliest_multi_bernoulli_outcomes import likeliest_multi_bernoulli_outcomes

# Solution from https://stackoverflow.com/a/66996523
__all_exports = [
    sorted_merge,
    smallest_subset_sums,
    likeliest_multi_bernoulli_outcomes,
]

for e in __all_exports:
    e.__module__ = __name__

__all__ = [e.__name__ for e in __all_exports]