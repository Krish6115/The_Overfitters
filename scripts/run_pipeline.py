#!/usr/bin/env python3
"""
End-to-End Multilingual Entity Matching Pipeline Runner
======================================================
Executes full lifecycle:
1. Data loading and multilingual normalization
2. Country-partitioned multi-route TF-IDF blocking
3. GroupKFold LightGBM training with hard-negative sampling
4. Macro F0.5 threshold optimization
5. Global 1-to-1 bipartite assignment
6. Formatted submission generation and provenance validation
"""

import os
import sys
import time
import argparse
import logging
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import Config
from src.preprocessing import preprocess_entity_dataframe, base_clean
from src.blocking import Blocker
from src.features import (
    build_entity_lookup,
    generate_candidate_features,
    extract_pairwise_features,
)
from src.model import EntityMatchingModel
from src.calibration import sweep_optimal_threshold, compute_macro_f05
from src.bipartite_matching import BipartiteMatchingSolver

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s | %(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("PipelineRunner")


def parse_args():
    parser = argparse.ArgumentParser(description="Multilingual Entity Matching Pipeline")
    parser.add_argument("--mode", type=str, default="full", choices=["smoke", "validation", "full"],
                        help="Execution mode (smoke=1k sample, validation=10%% subset, full=all data)")
    parser.add_argument("--dataset-root", type=str, default="", help="Path to student_resource directory")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory for predictions")
    parser.add_argument("--cache-dir", type=str, default="cache", help="Directory for temporary parquets")
    parser.add_argument("--threshold", type=float, default=None, help="Override decision threshold")
    return parser.parse_args()


def load_tsv(path: str) -> pd.DataFrame:
    logger.info("Loading TSV: %s", path)
    return pd.read_csv(path, sep="\t", dtype=str)


def parse_ground_truth(gt_path: str) -> Tuple[Dict[str, Set[str]], List[Tuple[str, str]]]:
    raw = load_tsv(gt_path)
    gt_dict: Dict[str, Set[str]] = {}
    pos_pairs: List[Tuple[str, str]] = []

    for _, row in raw.iterrows():
        s1 = str(row["source1_entity_id"]).strip()
        val = row.get("matched_entity_ids", "")
        if pd.isna(val) or str(val).strip() in ("", "nan"):
            gt_dict[s1] = set()
            continue
        ids = [x.strip() for x in str(val).split(",") if x.strip() and x.strip() != "nan"]
        gt_dict[s1] = set(ids)
        for mid in ids:
            pos_pairs.append((s1, mid))

    logger.info("Ground truth loaded: %d entities, %d positive pairs", len(gt_dict), len(pos_pairs))
    return gt_dict, pos_pairs


def main():
    args = parse_args()
    cfg = Config(
        mode=args.mode,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
    )

    logger.info("=== STAGE 1: Data Ingestion & Canonicalization ===")
    train_s1 = preprocess_entity_dataframe(load_tsv(cfg.train_s1_path))
    train_s2 = preprocess_entity_dataframe(load_tsv(cfg.train_s2_path))
    train_s3 = preprocess_entity_dataframe(load_tsv(cfg.train_s3_path))
    gt_dict, gt_pos_pairs = parse_ground_truth(cfg.train_gt_path)

    if cfg.mode == "smoke":
        train_s1 = train_s1.head(1000).reset_index(drop=True)
        train_s2 = train_s2.head(5000).reset_index(drop=True)
        train_s3 = train_s3.head(5000).reset_index(drop=True)
    elif cfg.mode == "validation":
        train_s1 = train_s1.sample(frac=0.10, random_state=42).reset_index(drop=True)

    gallery_train = pd.concat([train_s2, train_s3], ignore_index=True)
    logger.info("Train Gallery: %d entities", len(gallery_train))

    logger.info("=== STAGE 2: Multi-Route Candidate Blocking ===")
    blocker = Blocker(cfg)
    blocker.fit(gallery_train)
    train_candidates = blocker.generate_candidates(train_s1, cfg.cache_dir)
    logger.info("Generated %d candidate pairs for training", len(train_candidates))

    logger.info("=== STAGE 3: Pairwise Feature Engineering ===")
    s1_lookup = build_entity_lookup(train_s1)
    g_lookup = build_entity_lookup(gallery_train)

    # Attach Ground Truth Labels
    gt_pair_set = set(gt_pos_pairs)
    train_candidates["label"] = [
        1 if (str(s), str(m)) in gt_pair_set else 0
        for s, m in zip(train_candidates["s1_id"], train_candidates["matched_id"])
    ]
    train_features = generate_candidate_features(train_candidates, s1_lookup, g_lookup)
    train_features["label"] = train_candidates["label"].values

    logger.info("=== STAGE 4: GroupKFold LightGBM Ranking ===")
    model = EntityMatchingModel(cfg)
    groups = train_candidates["s1_id"].astype(str).values
    oof_probs, _ = model.train_cv(train_features, train_features["label"].values, groups)

    logger.info("=== STAGE 5: Macro F0.5 Threshold Optimization ===")
    all_train_s1 = set(train_s1["entity_id"].astype(str))
    best_t, best_score, _ = sweep_optimal_threshold(
        oof_probs,
        train_candidates["s1_id"].astype(str).values,
        train_candidates["matched_id"].astype(str).values,
        gt_dict,
        all_train_s1,
    )
    decision_threshold = args.threshold if args.threshold is not None else best_t
    logger.info("Selected Decision Threshold: %.3f", decision_threshold)

    logger.info("=== STAGE 6: Test Inference & Bipartite Dedup ===")
    test_s1 = preprocess_entity_dataframe(load_tsv(cfg.test_s1_path))
    test_s2 = preprocess_entity_dataframe(load_tsv(cfg.test_s2_path))
    test_s3 = preprocess_entity_dataframe(load_tsv(cfg.test_s3_path))
    gallery_test = pd.concat([test_s2, test_s3], ignore_index=True)

    blocker_test = Blocker(cfg)
    blocker_test.fit(gallery_test)
    test_candidates = blocker_test.generate_candidates(test_s1, cfg.cache_dir)

    s1_test_lookup = build_entity_lookup(test_s1)
    g_test_lookup = build_entity_lookup(gallery_test)
    test_features = generate_candidate_features(test_candidates, s1_test_lookup, g_test_lookup)

    test_candidates["prob"] = model.predict_proba(test_features)
    scored_test = test_candidates[test_candidates["prob"] >= decision_threshold].copy()

    solver = BipartiteMatchingSolver()
    matches_dict, _, _ = solver.solve_greedy(scored_test, score_col="prob")

    logger.info("=== STAGE 7: Writing Final Submissions ===")
    out_matching = os.path.join(cfg.output_dir, "matching_results.tsv")
    out_candidates = os.path.join(cfg.output_dir, "candidate_pairs.tsv")

    # Group candidates for candidate_pairs.tsv
    cand_by_s1 = test_candidates.groupby("s1_id")["matched_id"].apply(lambda s: list(s)).to_dict()

    all_test_s1_ids = test_s1["entity_id"].astype(str).values
    with open(out_matching, "w", encoding="utf-8") as fm, open(out_candidates, "w", encoding="utf-8") as fc:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in all_test_s1_ids:
            mids = matches_dict.get(sid, [])
            cids = cand_by_s1.get(sid, [])
            fm.write(f"{sid}\t{','.join(mids)}\n")
            fc.write(f"{sid}\t{','.join(cids)}\n")

    logger.info("Saved matching results to: %s", out_matching)
    logger.info("Saved candidate pairs to:  %s", out_candidates)
    logger.info("Pipeline run finished successfully.")


if __name__ == "__main__":
    main()
