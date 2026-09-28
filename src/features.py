"""
Multilingual Pairwise Feature Engineering Engine
================================================
Computes granular lexical, phonetic, structural, and geographic interaction
features for candidate entity pairs.
"""

import re
import logging
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

from src.preprocessing import (
    base_clean,
    canonicalize_business_name,
    canonicalize_address,
    extract_numbers,
)

logger = logging.getLogger(__name__)

# String distance backend resolution
try:
    from rapidfuzz.distance import Levenshtein as _Lev
    from rapidfuzz import fuzz

    def lev_ratio(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        return float(_Lev.normalized_similarity(a, b))

    def token_sort_ratio(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        return float(fuzz.token_sort_ratio(a, b))

    def token_set_ratio(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        return float(fuzz.token_set_ratio(a, b))

except ImportError:
    from difflib import SequenceMatcher

    def lev_ratio(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        return float(SequenceMatcher(None, a, b).ratio())

    def token_sort_ratio(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        sa = " ".join(sorted(a.split()))
        sb = " ".join(sorted(b.split()))
        return float(SequenceMatcher(None, sa, sb).ratio() * 100.0)

    def token_set_ratio(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        set_a = set(a.split())
        set_b = set(b.split())
        inter = " ".join(sorted(set_a & set_b))
        diff_a = " ".join(sorted(set_a - set_b))
        diff_b = " ".join(sorted(set_b - set_a))
        t1 = f"{inter} {diff_a}".strip()
        t2 = f"{inter} {diff_b}".strip()
        return float(max(
            SequenceMatcher(None, inter, t1).ratio(),
            SequenceMatcher(None, inter, t2).ratio(),
            SequenceMatcher(None, t1, t2).ratio()
        ) * 100.0)


def token_jaccard(a: str, b: str) -> float:
    """Computes whitespace-token Jaccard similarity."""
    if not a or not b:
        return 0.0
    sa, sb = set(a.split()), set(b.split())
    union = len(sa | sb)
    return len(sa & sb) / union if union else 0.0


def char_jaccard(a: str, b: str) -> float:
    """Computes character set Jaccard similarity."""
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    union = len(sa | sb)
    return len(sa & sb) / union if union else 0.0


def extract_features_for_pairs(
    s1_names: List[str],
    s1_addrs: List[str],
    s1_cnames: List[str],
    s1_caddrs: List[str],
    s1_countries: List[str],
    g_names: List[str],
    g_addrs: List[str],
    g_cnames: List[str],
    g_caddrs: List[str],
    g_countries: List[str],
    g_sources: List[str],
) -> pd.DataFrame:
    """
    Computes dense matrix of pairwise similarity features across query and candidate records.
    """
    n = len(s1_names)
    f: Dict[str, np.ndarray] = {}

    # 1. Levenshtein Metrics
    f["lev_name"] = np.array([lev_ratio(a, b) for a, b in zip(s1_names, g_names)], dtype=np.float32)
    f["lev_cname"] = np.array([lev_ratio(a, b) for a, b in zip(s1_cnames, g_cnames)], dtype=np.float32)
    f["lev_addr"] = np.array([lev_ratio(a, b) for a, b in zip(s1_caddrs, g_caddrs)], dtype=np.float32)

    # 2. Token Sort & Set Similarities (RapidFuzz scaled 0.0-1.0)
    f["tok_sort_cname"] = np.array(
        [token_sort_ratio(a, b) / 100.0 for a, b in zip(s1_cnames, g_cnames)],
        dtype=np.float32,
    )
    f["tok_set_cname"] = np.array(
        [token_set_ratio(a, b) / 100.0 for a, b in zip(s1_cnames, g_cnames)],
        dtype=np.float32,
    )
    f["tok_set_addr"] = np.array(
        [token_set_ratio(a, b) / 100.0 for a, b in zip(s1_caddrs, g_caddrs)],
        dtype=np.float32,
    )

    # 3. Jaccard Overlaps
    f["jac_cname"] = np.array([token_jaccard(a, b) for a, b in zip(s1_cnames, g_cnames)], dtype=np.float32)
    f["jac_addr"] = np.array([token_jaccard(a, b) for a, b in zip(s1_caddrs, g_caddrs)], dtype=np.float32)
    f["char_jac"] = np.array([char_jaccard(a, b) for a, b in zip(s1_cnames, g_cnames)], dtype=np.float32)

    # 4. Exact Matches
    f["exact_cname"] = (np.array(s1_cnames) == np.array(g_cnames)).astype(np.int8)
    f["country_match"] = (np.array(s1_countries) == np.array(g_countries)).astype(np.int8)

    # 5. Length & Token Discrepancies
    l1 = np.array([len(x) for x in s1_cnames], dtype=np.float32)
    l2 = np.array([len(x) for x in g_cnames], dtype=np.float32)
    max_l = np.maximum(l1, l2)
    f["name_len_ratio"] = np.where(max_l > 0, np.minimum(l1, l2) / max_l, 1.0).astype(np.float32)

    tok1 = np.array([len(x.split()) for x in s1_cnames], dtype=np.int16)
    tok2 = np.array([len(x.split()) for x in g_cnames], dtype=np.int16)
    f["name_tok_diff"] = np.abs(tok1 - tok2).astype(np.int8)

    # 6. Prefix Consistency
    f["prefix3"] = np.array([
        1 if len(a) >= 3 and len(b) >= 3 and a[:3] == b[:3] else 0
        for a, b in zip(s1_cnames, g_cnames)
    ], dtype=np.int8)
    f["prefix5"] = np.array([
        1 if len(a) >= 5 and len(b) >= 5 and a[:5] == b[:5] else 0
        for a, b in zip(s1_cnames, g_cnames)
    ], dtype=np.int8)

    # 7. Common Address Tokens
    f["common_addr_tok"] = np.array([
        len(set(a.split()) & set(b.split())) if a and b else 0
        for a, b in zip(s1_caddrs, g_caddrs)
    ], dtype=np.int16)

    # 8. Numeric & Postal Code Logic
    digit_jaccard = np.empty(n, dtype=np.float32)
    zip_conflict = np.zeros(n, dtype=np.int8)
    has_num_overlap = np.zeros(n, dtype=np.int8)

    for i in range(n):
        nums1 = extract_numbers(s1_caddrs[i])
        nums2 = extract_numbers(g_caddrs[i])
        if nums1 and nums2:
            inter = len(nums1 & nums2)
            union = len(nums1 | nums2)
            digit_jaccard[i] = inter / union if union else 0.0
            has_num_overlap[i] = 1 if inter > 0 else 0

            # Detect postal code conflict (5-6 digit numbers)
            z1 = {x for x in nums1 if len(x) in (5, 6)}
            z2 = {x for x in nums2 if len(x) in (5, 6)}
            if z1 and z2 and not (z1 & z2):
                zip_conflict[i] = 1
        else:
            digit_jaccard[i] = 0.0
            has_num_overlap[i] = 0

    f["digit_jaccard"] = digit_jaccard
    f["zip_conflict"] = zip_conflict
    f["has_num_overlap"] = has_num_overlap

    # 9. Source Indicators
    f["is_s2"] = np.array([1 if str(s).startswith("S2") else 0 for s in g_sources], dtype=np.int8)
    f["is_s3"] = np.array([1 if str(s).startswith("S3") else 0 for s in g_sources], dtype=np.int8)

    return pd.DataFrame(f)


def build_entity_lookup(df: pd.DataFrame) -> Dict[str, Tuple[str, str, str, str, str]]:
    """
    Builds optimized lookup dictionary mapping entity_id to:
    (norm_name, norm_addr, canon_name, canon_addr, country).
    """
    eids = df["entity_id"].astype(str).str.strip().values
    nns = df["norm_name"].values if "norm_name" in df else df["business_name"].apply(base_clean).values
    nas = df["norm_addr"].values if "norm_addr" in df else df["business_address"].apply(base_clean).values
    cns = df["canon_name"].values if "canon_name" in df else df["business_name"].apply(canonicalize_business_name).values
    cas = df["canon_addr"].values if "canon_addr" in df else df["business_address"].apply(canonicalize_address).values
    cos = df["country"].fillna("").astype(str).str.strip().values
    return dict(zip(eids, zip(nns, nas, cns, cas, cos)))


def generate_candidate_features(
    candidates_df: pd.DataFrame,
    s1_lookup: Dict[str, Tuple[str, str, str, str, str]],
    gallery_lookup: Dict[str, Tuple[str, str, str, str, str]],
) -> pd.DataFrame:
    """
    Joins candidate pairs with preprocessed entity data and computes complete feature table.
    """
    empty_tuple = ("", "", "", "", "")
    s1_ids = candidates_df["s1_id"].astype(str).values
    g_ids = candidates_df["matched_id"].astype(str).values

    s1n = [s1_lookup.get(s, empty_tuple)[0] for s in s1_ids]
    s1a = [s1_lookup.get(s, empty_tuple)[1] for s in s1_ids]
    s1cn = [s1_lookup.get(s, empty_tuple)[2] for s in s1_ids]
    s1ca = [s1_lookup.get(s, empty_tuple)[3] for s in s1_ids]
    s1co = [s1_lookup.get(s, empty_tuple)[4] for s in s1_ids]

    gn = [gallery_lookup.get(g, empty_tuple)[0] for g in g_ids]
    ga = [gallery_lookup.get(g, empty_tuple)[1] for g in g_ids]
    gcn = [gallery_lookup.get(g, empty_tuple)[2] for g in g_ids]
    gca = [gallery_lookup.get(g, empty_tuple)[3] for g in g_ids]
    gco = [gallery_lookup.get(g, empty_tuple)[4] for g in g_ids]

    feature_df = extract_features_for_pairs(
        s1n, s1a, s1cn, s1ca, s1co,
        gn, ga, gcn, gca, gco, g_ids
    )

    # Attach route scores if present in candidates_df
    if "name_score" in candidates_df.columns:
        feature_df["tfidf_name_score"] = candidates_df["name_score"].values.astype(np.float32)
    if "addr_score" in candidates_df.columns:
        feature_df["tfidf_addr_score"] = candidates_df["addr_score"].values.astype(np.float32)

    return feature_df
