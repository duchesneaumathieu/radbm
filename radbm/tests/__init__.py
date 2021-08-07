#================ loaders ================#
from .loaders_base import TestLoader, TestIRLoader
from .loaders_rss_conjunctive_boolean import TestConjunctiveBooleanRSS


#================ losses ================#
from .losses_binary_classification import TestFbetaLoss, TestBCELoss
from .losses_matching_bernoulli import TestFbetaMBMLoss, TestBCEMBMLoss
from .losses_matching_mihash import TestMIHashMatchingLoss
from .losses_matching_hashnet import TestHashNetMatchingLoss

#================  match  ================#
from .match_bernoulli_membership import TestMultiBernoulliMembershipMatch

#================ metrics ================#
from .metrics_hamming import TestHammingPRCurve
from .metrics_subset import TestSubsetDistance
from .metrics_user_cost import TestUCK
from .metrics_engine_metrics import TestTCR
from .metrics_pre_ap import TestPreAP


#================ search ================#
from .search_base import TestBaseSDS
from .search_heap import TestKeyValueHeap
from .search_dictionary import TestDictionarySearch
from .search_binary_hamming_multi_probing import TestHammingMultiProbing
from .search_binary_bernoulli_multi_probing import TestBernoulliMultiProbing
from .search_reduction_base import TestPointwiseReduction
from .search_reduction_hamming import TestHammingReduction
from .search_reduction_bernoulli import TestBernoulliReduction


#================ utils ================#
from .utils import Test_unique_list, TestRamp
from .utils_generators_sorted_merge import TestSortedMerge
from .utils_generators_smallest_subset_sums import TestSmallestSubsetSums
from .utils_generators_likeliest_multi_bernoulli_outcomes import TestLikeliestMultiBernoulliOutcomes
from .utils_fetch import Test_fetch_file
from .utils_numpy_function import TestNumpyFunction
from .utils_numpy_logical import TestNumpyLogical, TestGraphRepr
from .utils_numpy_random import (
    TestUniqueRandint,
    TestNoSubsetUniqueRandint,
    TestUniformNChooseK,
)
from .utils_torch import TorchCast
from .utils_torch_poisson_binomial import (
    TestLogPoissonBinomial,
    TestLogHammingBinomial,
)
from .utils_torch_multi_bernoulli_log_arithmetic import (
    TestLogAny,
    TestMultiBernoulliLogArithmetic,
)
from .utils_torch_multi_bernoulli_match import (
    TestHammingMatch,
    TestMultiIndexingMatch,
)
    
from .utils_torch_color import TestTorchColor
from .utils_stats import TestHypergeometric
from .utils_torch_regularization import TestHuberLoss
from .utils_time import TestChronometer
