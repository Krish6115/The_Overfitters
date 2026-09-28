"""
Multilingual Entity Resolution & Record Linkage Engine
======================================================
High-precision entity matching across heterogeneous data sources.
"""

from src.config import Config, default_config
from src.preprocessing import (
    base_clean,
    canonicalize_business_name,
    canonicalize_address,
    extract_numbers,
    is_non_latin,
    preprocess_entity_dataframe,
)
from src.blocking import Blocker, sparse_topk
from src.features import (
    lev_ratio,
    token_sort_ratio,
    token_set_ratio,
    token_jaccard,
    char_jaccard,
    extract_features_for_pairs,
    build_entity_lookup,
    generate_candidate_features,
)
from src.model import EntityMatchingModel
from src.calibration import (
    compute_macro_f05,
    sweep_optimal_threshold,
    CountryAwareCalibrator,
)
from src.bipartite_matching import BipartiteMatchingSolver
from src.postprocessing import PrecisionOptimizer

__version__ = "1.0.0"

__all__ = [
    "Config",
    "default_config",
    "base_clean",
    "canonicalize_business_name",
    "canonicalize_address",
    "extract_numbers",
    "is_non_latin",
    "preprocess_entity_dataframe",
    "Blocker",
    "sparse_topk",
    "lev_ratio",
    "token_sort_ratio",
    "token_set_ratio",
    "token_jaccard",
    "char_jaccard",
    "extract_features_for_pairs",
    "build_entity_lookup",
    "generate_candidate_features",
    "EntityMatchingModel",
    "compute_macro_f05",
    "sweep_optimal_threshold",
    "CountryAwareCalibrator",
    "BipartiteMatchingSolver",
    "PrecisionOptimizer",
]
