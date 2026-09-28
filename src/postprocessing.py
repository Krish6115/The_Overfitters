"""
Cross-Gallery Structural Completion & Singleton Protection Engine
==================================================================
Implements post-processing heuristics that optimize the precision-weighted F0.5 metric:
1. Cross-Gallery Completion: Rescues high-confidence missing sibling entities
   (e.g., S1 has S2 match but missed S3, or vice-versa) using dual-constraint verification.
2. Singleton Guard: Strictly protects unlinked singletons against false-positive pollution.
3. Global Deduplication: Enforces zero target collision across all recovered links.
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd

from src.features import token_sort_ratio, token_set_ratio
from src.preprocessing import extract_numbers, base_clean

logger = logging.getLogger(__name__)


class PrecisionOptimizer:
    """
    Precision-Max structural optimization engine.
    Elevates calibrated predictions to peak competition performance.
    """

    def __init__(
        self,
        min_name_sort_cross: float = 92.0,
        min_addr_set_cross: float = 65.0,
        singleton_name_threshold: float = 96.0,
        singleton_addr_threshold: float = 70.0,
    ):
        self.min_name_sort_cross = min_name_sort_cross
        self.min_addr_set_cross = min_addr_set_cross
        self.singleton_name_threshold = singleton_name_threshold
        self.singleton_addr_threshold = singleton_addr_threshold

    def optimize_matches(
        self,
        current_matches: Dict[str, List[str]],
        candidate_pool: Dict[str, List[str]],
        entity_metadata: Dict[str, Tuple[str, str, str, Set[str]]],
        scan_depth_cross: int = 12,
        scan_depth_singleton: int = 4,
    ) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
        """
        Executes precision optimization across candidates.

        Args:
            current_matches: Existing mapping of s1_id -> list of matched IDs.
            candidate_pool: Candidate pool mapping s1_id -> list of candidate IDs.
            entity_metadata: Mapping of eid -> (country, clean_name, clean_addr, street_nums).
            scan_depth_cross: Depth of candidates to evaluate for asymmetric completion.
            scan_depth_singleton: Depth of candidates to evaluate for singleton rescue.

        Returns:
            (optimized_matches, statistics)
        """
        # Identify currently claimed targets
        claimed_targets: Set[str] = set()
        for mids in current_matches.values():
            for m in mids:
                claimed_targets.add(m)

        # Categorize query status
        s2_only = {
            s1 for s1, mids in current_matches.items()
            if any(m.startswith("S2-") for m in mids) and not any(m.startswith("S3-") for m in mids)
        }
        s3_only = {
            s1 for s1, mids in current_matches.items()
            if any(m.startswith("S3-") for m in mids) and not any(m.startswith("S2-") for m in mids)
        }
        empty_queries = {
            s1 for s1, mids in current_matches.items() if not mids
        }

        logger.info(
            "Baseline queries: %d total | S2-only: %d | S3-only: %d | Empty singletons: %d",
            len(current_matches), len(s2_only), len(s3_only), len(empty_queries)
        )

        candidates_to_evaluate: List[Tuple[str, str, str]] = []

        for s1, cands in candidate_pool.items():
            if not cands:
                continue

            if s1 in empty_queries:
                c2_cnt = c3_cnt = 0
                for c in cands:
                    if c.startswith("S2-") and c not in claimed_targets and c2_cnt < scan_depth_singleton:
                        candidates_to_evaluate.append(("singleton", s1, c))
                        c2_cnt += 1
                    elif c.startswith("S3-") and c not in claimed_targets and c3_cnt < scan_depth_singleton:
                        candidates_to_evaluate.append(("singleton", s1, c))
                        c3_cnt += 1
                    if c2_cnt >= scan_depth_singleton and c3_cnt >= scan_depth_singleton:
                        break

            elif s1 in s2_only:
                cnt = 0
                for c in cands:
                    if c.startswith("S3-") and c not in claimed_targets:
                        candidates_to_evaluate.append(("cross_s3", s1, c))
                        cnt += 1
                        if cnt >= scan_depth_cross:
                            break

            elif s1 in s3_only:
                cnt = 0
                for c in cands:
                    if c.startswith("S2-") and c not in claimed_targets:
                        candidates_to_evaluate.append(("cross_s2", s1, c))
                        cnt += 1
                        if cnt >= scan_depth_cross:
                            break

        logger.info("Evaluating %d high-potential candidate links ...", len(candidates_to_evaluate))

        # Precision Verification
        valid_candidates: List[Tuple[float, str, str, str]] = []

        for cat, s1, cand in candidates_to_evaluate:
            if s1 not in entity_metadata or cand not in entity_metadata:
                continue

            c1, n1, a1, nums1 = entity_metadata[s1]
            c2, n2, a2, nums2 = entity_metadata[cand]

            # Country strictness
            if c1 and c2 and c1 != c2:
                continue

            name_sort = token_sort_ratio(n1, n2)
            name_set = token_set_ratio(n1, n2)
            addr_set = token_set_ratio(a1, a2)

            has_num_overlap = bool(nums1 & nums2) if (nums1 and nums2) else None
            composite_score = 0.60 * name_set + 0.25 * addr_set + 0.15 * name_sort

            if cat == "singleton":
                # Singleton Guard: strictly require near-perfect name and number alignment
                if has_num_overlap and name_sort >= self.singleton_name_threshold and addr_set >= self.singleton_addr_threshold:
                    valid_candidates.append((composite_score + 15.0, s1, cand, cat))
                elif has_num_overlap is None and name_sort >= 99.0 and addr_set >= 80.0:
                    valid_candidates.append((composite_score + 10.0, s1, cand, cat))

            elif "cross" in cat:
                # Cross-Gallery Completion on already partially matched entities
                if has_num_overlap:
                    if name_sort >= self.min_name_sort_cross and addr_set >= self.min_addr_set_cross:
                        valid_candidates.append((composite_score, s1, cand, cat))
                    elif name_set >= 95.0 and addr_set >= 70.0:
                        valid_candidates.append((composite_score, s1, cand, cat))
                elif has_num_overlap is None:
                    if name_sort >= 96.0 and addr_set >= 75.0:
                        valid_candidates.append((composite_score, s1, cand, cat))

        # Sort descending by composite quality score
        valid_candidates.sort(key=lambda x: x[0], reverse=True)

        # Global 1-to-1 Bipartite Deduplication
        accepted_by_s1: Dict[str, List[str]] = defaultdict(list)
        accepted_counts: Dict[str, int] = defaultdict(int)
        global_claimed = set(claimed_targets)

        for score, s1, cand, cat in valid_candidates:
            if cand in global_claimed:
                continue
            if len(accepted_by_s1[s1]) >= 2:
                continue

            global_claimed.add(cand)
            accepted_by_s1[s1].append(cand)
            accepted_counts[cat] += 1

        logger.info(
            "Accepted %d new ultra-precision links (Cross-S2: %d, Cross-S3: %d, Singleton-Rescue: %d)",
            sum(accepted_counts.values()),
            accepted_counts["cross_s2"],
            accepted_counts["cross_s3"],
            accepted_counts["singleton"],
        )

        # Merge with baseline matches
        optimized_matches: Dict[str, List[str]] = {}
        for s1, current_mids in current_matches.items():
            new_mids = accepted_by_s1.get(s1, [])
            merged = sorted(set(current_mids) | set(new_mids))
            optimized_matches[s1] = merged

        stats = {
            "total_new_links": sum(accepted_counts.values()),
            "cross_s2_recovered": accepted_counts["cross_s2"],
            "cross_s3_recovered": accepted_counts["cross_s3"],
            "singletons_rescued": accepted_counts["singleton"],
            "final_singletons": sum(1 for v in optimized_matches.values() if not v),
        }

        return optimized_matches, stats
