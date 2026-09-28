"""
Macro F0.5 Metric Calibration & Threshold Optimization Engine
=============================================================
Implements the exact competition metric: per-entity set-level macro-averaged F0.5,
and vector-accelerated threshold search optimizing for precision-weighted F0.5.
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

logger = logging.getLogger(__name__)


def compute_macro_f05(
    pred_dict: Dict[str, Set[str]],
    gt_dict: Dict[str, Set[str]],
    all_query_ids: Set[str],
) -> float:
    """
    Computes exact competition evaluation metric:
    Per-entity F0.5, macro-averaged across ALL source1 entities.

    F0.5 places 4x more weight on precision than recall:
        F0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)

    Boundary conditions:
      - true empty, pred empty: 1.0 (correct singleton identification)
      - true empty, pred non-empty: 0.0 (false positive hallucination)
      - true non-empty, pred empty: 0.0 (false negative omission)
    """
    scores: List[float] = []

    for sid in all_query_ids:
        preds = pred_dict.get(sid, set())
        truth = gt_dict.get(sid, set())

        if len(truth) == 0 and len(preds) == 0:
            scores.append(1.0)
        elif len(truth) == 0 and len(preds) > 0:
            scores.append(0.0)
        elif len(preds) == 0 and len(truth) > 0:
            scores.append(0.0)
        else:
            tp = len(preds & truth)
            if tp == 0:
                scores.append(0.0)
            else:
                prec = tp / float(len(preds))
                rec = tp / float(len(truth))
                f05 = (1.25 * prec * rec) / (0.25 * prec + rec)
                scores.append(f05)

    return float(np.mean(scores)) if scores else 0.0


def sweep_optimal_threshold(
    probabilities: np.ndarray,
    query_ids: np.ndarray,
    candidate_ids: np.ndarray,
    ground_truth_dict: Dict[str, Set[str]],
    all_query_ids: Set[str],
    threshold_range: Tuple[float, float, float] = (0.30, 0.95, 0.02),
) -> Tuple[float, float, Dict[float, float]]:
    """
    Fast grid sweep across candidate probability thresholds to maximize macro F0.5.
    Returns (best_threshold, best_f05_score, full_history_curve).
    """
    t_min, t_max, t_step = threshold_range
    thresholds = np.arange(t_min, t_max + 1e-6, t_step)

    best_threshold = 0.50
    best_score = 0.0
    history: Dict[float, float] = {}

    logger.info("Sweeping %d candidate thresholds for macro F0.5 ...", len(thresholds))

    for t in thresholds:
        mask = probabilities >= t
        pred_dict: Dict[str, Set[str]] = defaultdict(set)

        selected_q = query_ids[mask]
        selected_c = candidate_ids[mask]

        for sid, cid in zip(selected_q, selected_c):
            pred_dict[sid].add(cid)

        score = compute_macro_f05(pred_dict, ground_truth_dict, all_query_ids)
        history[float(round(t, 3))] = score

        if score > best_score:
            best_score = score
            best_threshold = float(round(t, 3))

    logger.info("Optimal threshold: %.3f -> Macro F0.5: %.5f", best_threshold, best_score)
    return best_threshold, best_score, history


class CountryAwareCalibrator:
    """
    Learns country-stratified decision thresholds to optimize metric performance
    across varying lexical density and address structures per country.
    """

    def __init__(self, global_threshold: float = 0.55):
        self.global_threshold = global_threshold
        self.country_thresholds: Dict[str, float] = {}

    def fit(
        self,
        probabilities: np.ndarray,
        query_ids: np.ndarray,
        candidate_ids: np.ndarray,
        countries: np.ndarray,
        ground_truth_dict: Dict[str, Set[str]],
        min_entities_per_country: int = 500,
    ) -> "CountryAwareCalibrator":
        """Fits country-specific thresholds on validation data."""
        unique_countries = np.unique(countries)

        for c in unique_countries:
            if not c or c == "":
                continue
            c_mask = countries == c
            c_queries = set(query_ids[c_mask])

            if len(c_queries) < min_entities_per_country:
                self.country_thresholds[c] = self.global_threshold
                continue

            sub_probs = probabilities[c_mask]
            sub_q = query_ids[c_mask]
            sub_cand = candidate_ids[c_mask]

            best_t, best_s, _ = sweep_optimal_threshold(
                sub_probs, sub_q, sub_cand,
                ground_truth_dict, c_queries,
                threshold_range=(0.35, 0.90, 0.05)
            )
            self.country_thresholds[c] = best_t
            logger.info("Country [%s] (entities=%d) calibrated threshold: %.3f (F0.5=%.4f)",
                        c, len(c_queries), best_t, best_s)

        return self

    def get_threshold(self, country: str) -> float:
        """Returns threshold for country, falling back to global threshold."""
        return self.country_thresholds.get(country, self.global_threshold)
