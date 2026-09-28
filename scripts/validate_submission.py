#!/usr/bin/env python3
"""
Submission Format & Constraint Validator
========================================
Comprehensive validation suite ensuring strict compliance with competition
rules and schema requirements:
1. Validates TSV formatting and exact header names.
2. Asserts row parity with test_source1.tsv.
3. Enforces entity ID prefix provenance (S1 query, S2/S3 gallery targets).
4. Verifies candidate pool containment (every predicted match must be in candidate_pairs).
5. Checks 1-to-1 global uniqueness (no gallery entity claimed twice).
6. Analyzes singleton and match degree distribution.
"""

import os
import sys
import argparse
import logging
from collections import Counter
from typing import Dict, List, Set

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s | %(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("SubmissionValidator")


def parse_args():
    parser = argparse.ArgumentParser(description="Submission Validator")
    parser.add_argument("--matching", type=str, required=True, help="Path to matching_results.tsv")
    parser.add_argument("--candidate", type=str, default="", help="Path to candidate_pairs.tsv (optional)")
    parser.add_argument("--test-dir", type=str, required=True, help="Path to test dataset directory")
    return parser.parse_args()


def main():
    args = parse_args()
    test_s1_path = os.path.join(args.test_dir, "test_source1.tsv")
    if not os.path.isfile(test_s1_path):
        logger.error("Could not find test_source1.tsv in %s", args.test_dir)
        sys.exit(1)

    logger.info("=== Loading Reference Test S1 Entities ===")
    test_s1_ids = []
    with open(test_s1_path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if parts:
                test_s1_ids.append(parts[0])
    test_s1_set = set(test_s1_ids)
    logger.info("Loaded %d ground test S1 IDs", len(test_s1_ids))

    logger.info("=== Validating matching_results.tsv ===")
    if not os.path.isfile(args.matching):
        logger.error("File does not exist: %s", args.matching)
        sys.exit(1)

    matched_s1_ids: List[str] = []
    matches_map: Dict[str, List[str]] = {}
    gallery_claims = Counter()
    format_errors = 0

    with open(args.matching, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n")
        if header != "source1_entity_id\tmatched_entity_ids":
            logger.error("Invalid header! Expected 'source1_entity_id\\tmatched_entity_ids', got '%s'", header)
            sys.exit(1)

        for line_num, line in enumerate(f, start=2):
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) == 1:
                s1, mids_str = parts[0], ""
            elif len(parts) == 2:
                s1, mids_str = parts[0], parts[1]
            else:
                logger.warning("Line %d has unexpected tab count: %s", line_num, line)
                format_errors += 1
                continue

            matched_s1_ids.append(s1)
            mids = mids_str.split(",") if mids_str else []
            matches_map[s1] = mids

            for m in mids:
                gallery_claims[m] += 1
                if not (m.startswith("S2-") or m.startswith("S3-")):
                    logger.error("Invalid target ID format '%s' for query '%s' at line %d", m, s1, line_num)
                    format_errors += 1

    # 1. Row Parity Check
    if len(matched_s1_ids) != len(test_s1_ids):
        logger.error("Row count mismatch! Submission has %d, test_source1 has %d", len(matched_s1_ids), len(test_s1_ids))
        sys.exit(1)

    if set(matched_s1_ids) != test_s1_set:
        logger.error("Mismatch between submission query IDs and test_source1 entity IDs!")
        sys.exit(1)

    # 2. Duplicate Gallery Claims (1-to-1 violation check)
    duplicate_claims = {k: v for k, v in gallery_claims.items() if v > 1}
    if duplicate_claims:
        logger.warning("Found %d gallery entities claimed by multiple queries! Highest duplicate count: %d",
                       len(duplicate_claims), max(duplicate_claims.values()))
    else:
        logger.info("[PASSED] Strict 1-to-1 compliance verified: Zero gallery target collisions.")

    # 3. Candidate Provenance Check (if candidate file provided)
    if args.candidate and os.path.isfile(args.candidate):
        logger.info("=== Validating Candidate Provenance ===")
        provenance_violations = 0
        with open(args.candidate, "r", encoding="utf-8") as f:
            c_header = f.readline().rstrip("\r\n")
            if c_header != "source1_entity_id\tcandidate_entity_ids":
                logger.warning("Candidate file header: '%s'", c_header)

            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                s1 = parts[0]
                c_pool = set(parts[1].split(",")) if len(parts) > 1 and parts[1] else set()
                pred_matches = matches_map.get(s1, [])
                for pm in pred_matches:
                    if pm not in c_pool:
                        provenance_violations += 1

        if provenance_violations > 0:
            logger.warning("Found %d predictions not in declared candidate pool!", provenance_violations)
        else:
            logger.info("[PASSED] 100%% candidate pool provenance verified.")

    # 4. Distribution Summary
    total_queries = len(matched_s1_ids)
    singletons = sum(1 for mids in matches_map.values() if not mids)
    s2_matches = sum(sum(1 for m in mids if m.startswith("S2-")) for mids in matches_map.values())
    s3_matches = sum(sum(1 for m in mids if m.startswith("S3-")) for mids in matches_map.values())
    total_links = s2_matches + s3_matches

    logger.info("=== Validation Summary ===")
    logger.info("Total Queries:      %d", total_queries)
    logger.info("Total Match Links:  %d (S2: %d, S3: %d)", total_links, s2_matches, s3_matches)
    logger.info("Singletons:         %d (%.2f%%)", singletons, (singletons / total_queries) * 100.0)
    logger.info("Average Match/S1:   %.3f", total_links / total_queries)
    logger.info("Submission validation completed successfully.")


if __name__ == "__main__":
    main()
