import heapq
import numpy as np

class Subset(object):
    def __init__(self, items, total):
        self.items = items
        self.total = total
    
    def __hash__(self):
        return hash(tuple(self.items))
    
    def __eq__(self, subset):
        return self.items == subset.items
    
    def __lt__(self, subset):
        return self.total < subset.total
    
def next_subsets_generator(items, total, values):
    if not items or items[0] != 0:
        yield Subset([0]+items, total + values[0])
    for n, i in enumerate(items[:-1]):
        if i+1 < items[n+1]:
            yield Subset(
                items[:n]+[i+1]+items[n+1:],
                total - values[i] + values[i+1]
            )
    if items:
        i = items[-1]
        if i+1 < len(values):
            yield Subset(
                items[:-1]+[i+1],
                total - values[i] + values[i+1]
            )
        
def least_k_subsortedset_sum_generator(values, k=None):
    if any(v < 0 for v in values): raise ValueError('values must be positive')
    if k is None: k = 2**len(values)
    k = min(k, 2**len(values))
    heap = [Subset([], 0)]
    produced = set()
    for _ in range(k):
        subset = heapq.heappop(heap)
        yield subset.items
        for new_subset in next_subsets_generator(subset.items, subset.total, values):
            if new_subset not in produced:
                heapq.heappush(heap, new_subset)
                produced.add(new_subset)

def least_k_subset_sum_generator(values, k=None):
    perm = list(np.argsort(values))
    values = [values[p] for p in perm]
    for items in least_k_subsortedset_sum_generator(values, k):
        yield tuple(perm[i] for i in items)
        
def poisson_binomial_top_k_generator(log_probs, k=None):
    log_probs = np.array(log_probs)
    top1 = [0 for i in range(len(log_probs))]
    values = np.log(1-np.exp(log_probs))-log_probs
    flips = np.where(np.log(.5) < np.array(log_probs))[0]
    for i in flips:
        top1[i] = 1-top1[i]
        values[i] = -values[i]
    for subset in least_k_subset_sum_generator(values, k):
        bits = list(top1)
        for i in subset: bits[i] = 1-bits[i]
        yield tuple(bits)