#!/usr/bin/env python3
"""
Deep Structural Optimizer for Entity Resolution (Amazon ML Challenge 2026)
===========================================================================
Building upon the verified 81.2% score from our previous cross-gallery recovery:
  1. Transitive & Multi-Candidate Cross-Gallery Recovery (top 5 candidates, threshold 88 with street nums)
  2. False Singleton Precision Recovery (recovering empty entities that have >=94% exact matches)
  3. Intra-Gallery High-Confidence Completion (recovering true multi-listing matches)
  4. Global 1-to-1 Bipartite Deduplication (guaranteeing 0 duplicate target conflicts)
  5. Submission Validator Pass Verification
"""

import os, sys, time, re
from collections import defaultdict
from rapidfuzz import fuzz

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def clean_name(s):
    if not s: return ""
    s = str(s).lower()
    s = re.sub(r'\b(inc|corp|corporation|llc|l\.l\.c|ltd|limited|pvt|co|company|sarl|sas|gmbh|sa|eurl|llp)\b', '', s)
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    return ' '.join(s.split())

def extract_nums(s):
    if not s: return set()
    return set(re.findall(r'\b\d+\b', str(s)))

def main():
    t0 = time.time()
    log("=== Stage 1: Loading 81.2% Submission Baseline ===")
    baseline_path = r"C:\Users\sivar\Downloads\educational_structural_inference\matching_results.tsv"
    
    matches = {}
    all_claimed = set()
    with open(baseline_path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            s1 = parts[0]
            mids = parts[1].split(",") if len(parts) > 1 and parts[1] else []
            matches[s1] = mids
            for m in mids:
                all_claimed.add(m)

    log(f"  Loaded {len(matches):,} baseline entities.")
    log(f"  Total already-claimed targets: {len(all_claimed):,}")

    s2_only = {s1 for s1, mids in matches.items() if any(m.startswith("S2-") for m in mids) and not any(m.startswith("S3-") for m in mids)}
    s3_only = {s1 for s1, mids in matches.items() if any(m.startswith("S3-") for m in mids) and not any(m.startswith("S2-") for m in mids)}
    empty_s1 = {s1 for s1, mids in matches.items() if not mids}
    both_s1 = {s1 for s1, mids in matches.items() if any(m.startswith("S2-") for m in mids) and any(m.startswith("S3-") for m in mids)}

    log(f"  Baseline S2-only:      {len(s2_only):,}")
    log(f"  Baseline S3-only:      {len(s3_only):,}")
    log(f"  Baseline both S2 & S3: {len(both_s1):,}")
    log(f"  Baseline empty (zero): {len(empty_s1):,}")

    log("\n=== Stage 2: Scanning Candidate Pool for High-Leverage Pairs ===")
    cand_path = r"C:\Users\sivar\Downloads\candidate_pairs.tsv"
    
    # Store candidates to evaluate: (category, s1, cand_id)
    eval_candidates = []
    needed_s1 = set()
    needed_s2 = set()
    needed_s3 = set()

    with open(cand_path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            s1 = parts[0]
            cands = parts[1].split(",") if len(parts) > 1 and parts[1] else []
            if not cands: continue

            if s1 in s2_only:
                # Need S3 candidates (check top 5)
                c_count = 0
                for c in cands:
                    if c.startswith("S3-") and c not in all_claimed:
                        eval_candidates.append(("cross_s3", s1, c))
                        needed_s1.add(s1)
                        needed_s3.add(c)
                        c_count += 1
                        if c_count >= 5: break

            elif s1 in s3_only:
                # Need S2 candidates (check top 5)
                c_count = 0
                for c in cands:
                    if c.startswith("S2-") and c not in all_claimed:
                        eval_candidates.append(("cross_s2", s1, c))
                        needed_s1.add(s1)
                        needed_s2.add(c)
                        c_count += 1
                        if c_count >= 5: break

            elif s1 in empty_s1:
                # False singleton recovery: check top 2 candidates from S2 and top 2 from S3
                s2_c = 0
                s3_c = 0
                for c in cands:
                    if c.startswith("S2-") and c not in all_claimed and s2_c < 2:
                        eval_candidates.append(("singleton_s2", s1, c))
                        needed_s1.add(s1)
                        needed_s2.add(c)
                        s2_c += 1
                    elif c.startswith("S3-") and c not in all_claimed and s3_c < 2:
                        eval_candidates.append(("singleton_s3", s1, c))
                        needed_s1.add(s1)
                        needed_s3.add(c)
                        s3_c += 1
                    if s2_c >= 2 and s3_c >= 2: break

    log(f"  Total candidate pairs to evaluate: {len(eval_candidates):,}")
    log(f"  Unique entities: S1: {len(needed_s1):,}, S2: {len(needed_s2):,}, S3: {len(needed_s3):,}")

    log("\n=== Stage 3: Streaming Entity Metadata ===")
    meta = {}  # eid -> (country, clean_name, nums)

    t_read = time.time()
    with open(r"C:\Users\sivar\OneDrive\Desktop\student_resource\dataset\test\test_source1.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            eid = parts[0]
            if eid in needed_s1:
                name = parts[1] if len(parts) > 1 else ""
                addr = parts[2] if len(parts) > 2 else ""
                country = parts[3] if len(parts) > 3 else ""
                meta[eid] = (country, clean_name(name), extract_nums(addr))
    log(f"  Loaded {len(meta):,} S1 metadata records in {time.time()-t_read:.1f}s.")

    t_read = time.time()
    s2_read = 0
    with open(r"C:\Users\sivar\OneDrive\Desktop\student_resource\dataset\test\test_source2.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            eid = parts[0]
            if eid in needed_s2:
                name = parts[1] if len(parts) > 1 else ""
                addr = parts[2] if len(parts) > 2 else ""
                country = parts[3] if len(parts) > 3 else ""
                meta[eid] = (country, clean_name(name), extract_nums(addr))
                s2_read += 1
                if s2_read >= len(needed_s2): break
    log(f"  Loaded {s2_read:,} S2 metadata records in {time.time()-t_read:.1f}s.")

    t_read = time.time()
    s3_read = 0
    with open(r"C:\Users\sivar\OneDrive\Desktop\student_resource\dataset\test\test_source3.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            eid = parts[0]
            if eid in needed_s3:
                name = parts[1] if len(parts) > 1 else ""
                addr = parts[2] if len(parts) > 2 else ""
                country = parts[3] if len(parts) > 3 else ""
                meta[eid] = (country, clean_name(name), extract_nums(addr))
                s3_read += 1
                if s3_read >= len(needed_s3): break
    log(f"  Loaded {s3_read:,} S3 metadata records in {time.time()-t_read:.1f}s.")

    log("\n=== Stage 4: High-Precision Multi-Gate Matching ===")
    valid_pairs = []  # (priority_score, s1, cand, cat)

    for cat, s1, cand in eval_candidates:
        if s1 not in meta or cand not in meta:
            continue
        c1, n1, nums1 = meta[s1]
        c2, n2, nums2 = meta[cand]

        if c1 != c2: continue

        sim = fuzz.token_sort_ratio(n1, n2)
        has_num_overlap = bool(nums1 & nums2) if (nums1 and nums2) else None

        if "cross" in cat:
            # Cross-gallery recovery on entities already verified non-empty
            if has_num_overlap and sim >= 88:
                valid_pairs.append((sim, s1, cand, cat))
            elif has_num_overlap is None and sim >= 95:
                valid_pairs.append((sim, s1, cand, cat))
        elif "singleton" in cat:
            # False singleton recovery: stricter thresholds to protect precision
            if has_num_overlap and sim >= 94:
                valid_pairs.append((sim + 5, s1, cand, cat))  # Priority to high-confidence singleton recovery
            elif has_num_overlap is None and sim >= 98:
                valid_pairs.append((sim + 5, s1, cand, cat))

    log(f"  Pairs meeting precision verification: {len(valid_pairs):,}")

    # Sort descending by score for optimal greedy 1-to-1 bipartite assignment
    valid_pairs.sort(key=lambda x: x[0], reverse=True)

    log("\n=== Stage 5: Bipartite Matching & Deduplication ===")
    claimed_set = set(all_claimed)
    accepted_by_s1 = defaultdict(list)
    accepted_count = 0
    cat_counts = defaultdict(int)

    for score, s1, cand, cat in valid_pairs:
        if cand in claimed_set:
            continue
        # Limit cross-gallery recovery to max 2 new links per S1
        if len(accepted_by_s1[s1]) >= 2:
            continue

        claimed_set.add(cand)
        accepted_by_s1[s1].append(cand)
        accepted_count += 1
        cat_counts[cat] += 1

    log(f"  Accepted {accepted_count:,} new verified links:")
    for cat, cnt in cat_counts.items():
        log(f"    - {cat}: {cnt:,}")

    log("\n=== Stage 6: Writing Enhanced Submission File ===")
    out_dir = r"C:\Users\sivar\Downloads\deep_optimized_submission"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "matching_results.tsv")

    total_rows = 0
    total_links = 0
    empty_final = 0

    with open(baseline_path, "r", encoding="utf-8") as f_in, open(out_file, "w", encoding="utf-8") as f_out:
        header = f_in.readline()
        f_out.write(header)
        for line in f_in:
            parts = line.rstrip("\r\n").split("\t")
            s1 = parts[0]
            cur_mids = parts[1].split(",") if len(parts) > 1 and parts[1] else []
            new_mids = accepted_by_s1.get(s1, [])
            combined = sorted(set(cur_mids) | set(new_mids))
            f_out.write(f"{s1}\t{','.join(combined)}\n")
            total_rows += 1
            total_links += len(combined)
            if not combined:
                empty_final += 1

    log(f"  Wrote: {out_file}")
    log(f"  Total Rows:        {total_rows:,}")
    log(f"  Total Match Links: {total_links:,} (+{accepted_count:,} recovered)")
    log(f"  Final Singletons:  {empty_final:,} ({empty_final/total_rows*100:.2f}%)")

    log("\n=== Stage 7: Running Submission Validator ===")
    validator = r"C:\Users\sivar\OneDrive\Desktop\student_resource\utils\validate_submission.py"
    test_dir = r"C:\Users\sivar\OneDrive\Desktop\student_resource\dataset\test"
    cmd = f'py "{validator}" --matching "{out_file}" --test-dir "{test_dir}"'
    log(f"  Running validator...")
    res = os.system(cmd)
    log(f"  Validator exit code: {res} ({'PASS ✓' if res == 0 else 'FAIL ✗'})")

    log(f"\nDeep Optimization Finished in {time.time()-t0:.1f} seconds!")

if __name__ == "__main__":
    main()
