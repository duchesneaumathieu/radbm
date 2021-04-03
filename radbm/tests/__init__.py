from .utils import Test_unique_list, TestRamp
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
from .utils_torch_multi_bernoulli_match import TestHammingMatch
    
from .utils_torch_color import TestTorchColor
from .utils_stats import (
    Test_least_k_subset_sum_generator,
    Test_greatest_k_multi_bernoulli_outcomes_generator,
    TestHypergeometric,
)
from .utils_time import TestChronometer
from .loaders_base import TestLoader, TestIRLoader
from .loaders_rss_conjunctive_boolean import TestConjunctiveBooleanRSS
from .search_base import TestBaseSDS
from .search_mbsds import TestHashingMultiBernoulliSDS
from .search_gridsearch import TestGridSearch
from .search_radius import TestHammingRadiusSDS
from .search_elba_base import TestEfficientLearnableBinaryAccess
from .search_elba_fbeta import TestFbeta
from .search_elba_hbkl import TestHBKL
from .search_elba_mihash import TestMIHash
from .search_elba_hashnet import TestHashNet
from .metrics_oracle import TestOracleMetric
from .metrics_sswr import TestSSWR
from .metrics_hamming import TestHammingPRCurve
from .metrics_user_cost import TestUCK
from .metrics_pre_ap import TestPreAP
from .losses_binary_classification import TestFbetaLoss, TestBCELoss
from .losses_multi_bernoulli_matching import TestFbetaMBMLoss, TestBCEMBMLoss
