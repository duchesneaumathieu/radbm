from .user_cost import (
    user_cost_at_k_original,
    user_cost_at_k_from_counts,
    user_cost_at_k,
    user_cost_at_k_from_scores,
)

from .engine_metrics import (
    costs_at_k,
    total_cost_at_k,
    total_cost_ratio_from_costs,
    total_cost_ratio,
)

# This lets Sphinx know you want to document binary.hamming_multi_probing.HammingMultiProbing as binary.HammingMultiProbing.
# Solution from https://stackoverflow.com/a/66996523
__all_exports = [
    user_cost_at_k_original,
    user_cost_at_k_from_counts,
    user_cost_at_k,
    user_cost_at_k_from_scores,
    costs_at_k,
    total_cost_at_k,
    total_cost_ratio_from_costs,
    total_cost_ratio,
]

for e in __all_exports:
    e.__module__ = __name__

__all__ = [e.__name__ for e in __all_exports]