#!/usr/bin/env python3
"""
Educational Cross-Gallery Bipartite Inference & Candidate Recovery
===================================================================
Purpose:
  Demonstrate structural inference on entity resolution graphs:
  In ground truth, 80.5% of entities match BOTH Source 2 and Source 3.
  Our 0.811 baseline has ~295k entities with matches in S2 but 0 in S3,
  and ~326k entities with matches in S3 but 0 in S2.
  
  This script uses high-precision canonical string matching, street-number
  invariance, and strict 1-to-1 bipartite assignment to recover true missing
  cross-gallery links without harming precision or touching singletons.
"""

import os, sys, time, re
from collections import defaultdict
from rapidfuzz import fuzz

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def clean_name(s):
    if not s: return ""
    s = str(s).lower()
    # Strip common legal suffixes across US, India, France
    s = re.sub(r'\b(inc|corp|corporation|llc|l\.l\.c|ltd|limited|pvt|co|company|sarl|sas|gmbh|sa|eurl|llp)\b', '', s)
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    return ' '.join(s.split())

def extract_nums(s):
    if not s: return set()
    return set(re.findall(r'\b\d+\b', str(s)))

def main():
    t0 = time.time()
    log("=== Stage 1: Loading Baseline matching_results.tsv (0.811 peak) ===")
    baseline_path = r"C:\Users\sivar\Downloads\improved_output\matching_results.tsv"
    
    matches = {}
    all_claimed = set()
    with open(baseline_path, "r", encoding="utf-8") as f:
        header = f.readline()
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
    singletons = {s1 for s1, mids in matches.items() if not mids}
    log(f"  Baseline S2-only entities:  {len(s2_only):,}")
    log(f"  Baseline S3-only entities:  {len(s3_only):,}")
    log(f"  Baseline singletons:        {len(singletons):,} (Strictly protected, 0 edits)")

    log("\n=== Stage 2: Finding Candidates for Cross-Gallery Completion ===")
    cand_path = r"C:\Users\sivar\Downloads\candidate_pairs.tsv"
    needed_s1 = s2_only | s3_only
    
    # Store list of (s1, cand_id) pairs to evaluate
    eval_pairs = []
    needed_s1_meta = set()
    needed_s2_meta = set()
    needed_s3_meta = set()

    with open(cand_path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            s1 = parts[0]
            if s1 not in needed_s1:
                continue
            cands = parts[1].split(",") if len(parts) > 1 and parts[1] else []
            if s1 in s2_only:
                # Find top 2 unclaimed S3 candidates
                c_count = 0
                for c in cands:
                    if c.startswith("S3-") and c not in all_claimed:
                        eval_pairs.append((s1, c))
                        needed_s1_meta.add(s1)
                        needed_s3_meta.add(c)
                        c_count += 1
                        if c_count >= 2: break
            elif s1 in s3_only:
                # Find top 2 unclaimed S2 candidates
                c_count = 0
                for c in cands:
                    if c.startswith("S2-") and c not in all_claimed:
                        eval_pairs.append((s1, c))
                        needed_s1_meta.add(s1)
                        needed_s2_meta.add(c)
                        c_count += 1
                        if c_count >= 2: break

    log(f"  Generated {len(eval_pairs):,} high-potential candidate pairs to evaluate.")
    log(f"  Unique entities to fetch: S1: {len(needed_s1_meta):,}, S2: {len(needed_s2_meta):,}, S3: {len(needed_s3_meta):,}")

    log("\n=== Stage 3: Streaming Metadata for Evaluation Sets ===")
    meta = {}  # id -> (country, clean_name, nums)

    t_read = time.time()
    s1_path = r"C:\Users\sivar\OneDrive\Desktop\student_resource\dataset\test\test_source1.tsv"
    with open(s1_path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            eid = parts[0]
            if eid in needed_s1_meta:
                name = parts[1] if len(parts) > 1 else ""
                addr = parts[2] if len(parts) > 2 else ""
                country = parts[3] if len(parts) > 3 else ""
                meta[eid] = (country, clean_name(name), extract_nums(addr))
    log(f"  Loaded {len(meta):,} S1 records in {time.time()-t_read:.1f}s.")

    t_read = time.time()
    s2_path = r"C:\Users\sivar\OneDrive\Desktop\student_resource\dataset\test\test_source2.tsv"
    s2_count = 0
    with open(s2_path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            eid = parts[0]
            if eid in needed_s2_meta:
                name = parts[1] if len(parts) > 1 else ""
                addr = parts[2] if len(parts) > 2 else ""
                country = parts[3] if len(parts) > 3 else ""
                meta[eid] = (country, clean_name(name), extract_nums(addr))
                s2_count += 1
                if s2_count >= len(needed_s2_meta): break
    log(f"  Loaded {s2_count:,} S2 records in {time.time()-t_read:.1f}s.")

    t_read = time.time()
    s3_path = r"C:\Users\sivar\OneDrive\Desktop\student_resource\dataset\test\test_source3.tsv"
    s3_count = 0
    with open(s3_path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            eid = parts[0]
            if eid in needed_s3_meta:
                name = parts[1] if len(parts) > 1 else ""
                addr = parts[2] if len(parts) > 2 else ""
                country = parts[3] if len(parts) > 3 else ""
                meta[eid] = (country, clean_name(name), extract_nums(addr))
                s3_count += 1
                if s3_count >= len(needed_s3_meta): break
    log(f"  Loaded {s3_count:,} S3 records in {time.time()-t_read:.1f}s.")

    log("\n=== Stage 4: High-Precision Verification & Scoring ===")
    # Candidate quality scoring
    valid_candidates = []  # (score, s1, cand_id)

    for s1, cand in eval_pairs:
        if s1 not in meta or cand not in meta:
            continue
        c1, n1, nums1 = meta[s1]
        c2, n2, nums2 = meta[cand]

        # 1. Country MUST match exactly
        if c1 != c2:
            continue

        # 2. String similarity on legal-normalized name
        name_sim = fuzz.token_sort_ratio(n1, n2)

        # 3. Street numbers check
        if nums1 and nums2:
            has_num_overlap = bool(nums1 & nums2)
            # If street numbers match, require high name similarity (>= 90)
            if has_num_overlap and name_sim >= 90:
                valid_candidates.append((name_sim, s1, cand))
        else:
            # If street numbers are missing from one or both, require near-perfect name (>= 98)
            if name_sim >= 98:
                valid_candidates.append((name_sim, s1, cand))

    log(f"  Found {len(valid_candidates):,} candidates passing ultra-strict precision filters.")

    # Sort candidates by descending confidence score for greedy 1-to-1 bipartite assignment
    valid_candidates.sort(key=lambda x: x[0], reverse=True)

    log("\n=== Stage 5: Solving 1-to-1 Bipartite Matching on Recovered Pairs ===")
    accepted_assignments = []
    claimed_in_recovery = set(all_claimed)
    s1_recovered = set()

    for score, s1, cand in valid_candidates:
        if cand in claimed_in_recovery:
            continue
        # Limit recovery to at most 1 new cross-gallery match per S1 entity
        if s1 in s1_recovered:
            continue
        
        claimed_in_recovery.add(cand)
        s1_recovered.add(s1)
        accepted_assignments.append((s1, cand, score))

    log(f"  Accepted {len(accepted_assignments):,} new high-precision cross-gallery links.")
    log(f"  Examples of recovered links:")
    for s1, cand, score in accepted_assignments[:5]:
        _, n1, _ = meta[s1]
        _, n2, _ = meta[cand]
        log(f"    [{score}%] {s1} ('{n1}') <---> {cand} ('{n2}')")

    log("\n=== Stage 6: Exporting Enhanced matching_results.tsv ===")
    out_dir = r"C:\Users\sivar\Downloads\educational_structural_inference"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "matching_results.tsv")

    # Map recovered matches
    rec_map = defaultdict(list)
    for s1, cand, _ in accepted_assignments:
        rec_map[s1].append(cand)

    total_rows = 0
    total_links = 0
    with open(baseline_path, "r", encoding="utf-8") as f_in, open(out_file, "w", encoding="utf-8") as f_out:
        header = f_in.readline()
        f_out.write(header)
        for line in f_in:
            parts = line.rstrip("\r\n").split("\t")
            s1 = parts[0]
            cur_mids = parts[1].split(",") if len(parts) > 1 and parts[1] else []
            new_mids = rec_map.get(s1, [])
            combined = sorted(set(cur_mids) | set(new_mids))
            f_out.write(f"{s1}\t{','.join(combined)}\n")
            total_rows += 1
            total_links += len(combined)

    log(f"  Wrote: {out_file}")
    log(f"  Total Rows: {total_rows:,}")
    log(f"  Total Links: {total_links:,} (+{len(accepted_assignments):,} recovered)")

    log("\n=== Stage 7: Running Submission Validator ===")
    validator = r"C:\Users\sivar\OneDrive\Desktop\student_resource\utils\validate_submission.py"
    test_dir = r"C:\Users\sivar\OneDrive\Desktop\student_resource\dataset\test"
    cmd = f'py "{validator}" --matching "{out_file}" --test-dir "{test_dir}"'
    log(f"  Running: {cmd}")
    os.system(cmd)

    log(f"\nEducational Pipeline Completed in {time.time()-t0:.1f} seconds!")

if __name__ == "__main__":
    main()
