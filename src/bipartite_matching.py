"""
Capacitated Bipartite Matching & Target Deduplication Engine
============================================================
Solves the global 1-to-1 gallery assignment problem.
Guarantees that no gallery record (S2 or S3) is claimed by more than one
query entity (S1), while enforcing capacity constraints (at most 1 S2 and 1 S3 per S1).
Eliminates many-to-one false-positive collisions that severely penalize F0.5.
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd

logger = logging.getLogger(__name__)


class BipartiteMatchingSolver:
    """
    Solves global 1-to-1 maximum-weight bipartite assignment with structural capacity limits.
    """

    def __init__(
        self,
        max_s2_per_s1: int = 1,
        max_s3_per_s1: int = 1,
        max_total_per_s1: int = 2,
    ):
        self.max_s2 = max_s2_per_s1
        self.max_s3 = max_s3_per_s1
        self.max_total = max_total_per_s1

    def solve_greedy(
        self,
        scored_pairs_df: pd.DataFrame,
        s1_col: str = "s1_id",
        target_col: str = "matched_id",
        score_col: str = "prob",
        existing_claims: Optional[Set[str]] = None,
    ) -> Tuple[Dict[str, List[str]], Set[str], int]:
        """
        Greedy 1-to-1 assignment ordered by confidence score descending.

        Args:
            scored_pairs_df: DataFrame containing candidate pairs and confidence scores.
            s1_col: Column name for query ID.
            target_col: Column name for target gallery entity ID.
            score_col: Column name for confidence score / probability.
            existing_claims: Optional set of target IDs already claimed in previous stages.

        Returns:
            (matches_by_s1, all_claimed_targets, num_conflicts_resolved)
        """
        if scored_pairs_df.empty:
            return {}, set(existing_claims or set()), 0

        # Sort strictly descending by confidence
        df_sorted = scored_pairs_df.sort_values(by=score_col, ascending=False)

        claimed_targets: Set[str] = set(existing_claims) if existing_claims else set()
        matches_by_s1: Dict[str, List[str]] = defaultdict(list)
        s1_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"S2": 0, "S3": 0})
        conflicts = 0

        s1_arr = df_sorted[s1_col].astype(str).values
        target_arr = df_sorted[target_col].astype(str).values

        for s1, target in zip(s1_arr, target_arr):
            # Enforce global 1-to-1: target can only be assigned once globally
            if target in claimed_targets:
                conflicts += 1
                continue

            # Determine target gallery source
            source_type = "S2" if target.startswith("S2") else "S3"

            # Check capacity constraints for query entity
            current_counts = s1_counts[s1]
            total_matches = current_counts["S2"] + current_counts["S3"]

            if total_matches >= self.max_total:
                continue

            if source_type == "S2" and current_counts["S2"] >= self.max_s2:
                continue
            if source_type == "S3" and current_counts["S3"] >= self.max_s3:
                continue

            # Assign match
            claimed_targets.add(target)
            matches_by_s1[s1].append(target)
            current_counts[source_type] += 1

        logger.info(
            "Bipartite solver assigned %d matches across %d queries (resolved %d target collisions).",
            sum(len(v) for v in matches_by_s1.values()),
            len(matches_by_s1),
            conflicts,
        )

        return dict(matches_by_s1), claimed_targets, conflicts

    def deduplicate_dataframe(
        self,
        scored_pairs_df: pd.DataFrame,
        target_col: str = "matched_id",
        score_col: str = "prob",
    ) -> pd.DataFrame:
        """
        Deduplicates candidate pairs by keeping only the highest-scoring query per target entity.
        """
        if scored_pairs_df.empty:
            return scored_pairs_df

        df_sorted = scored_pairs_df.sort_values(by=score_col, ascending=False)
        deduped = df_sorted.drop_duplicates(subset=[target_col], keep="first").copy()
        logger.info(
            "Target deduplication: %d -> %d pairs (%d redundant claims removed).",
            len(scored_pairs_df),
            len(deduped),
            len(scored_pairs_df) - len(deduped),
        )
        return deduped
