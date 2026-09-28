#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 — RESCORER v2
========================================
SKIPS the 2-hour blocking step entirely.
Reuses existing candidate_pairs.tsv, then:
  1. Legal-suffix canonicalization (Corp→"", Pvt Ltd→"", SARL→"", etc.)
  2. 18 rich features (4× Levenshtein variants, digit overlap, zip conflict…)
  3. LightGBM trained on FULL ground truth (not 15% sample)
  4. Precision-biased threshold sweep (macro F0.5)
  5. Greedy 1-to-1 target assignment (each S2/S3 → at most 1 S1)
  6. Saves raw probabilities so you can re-threshold locally

Expected runtime: ~15–20 min on z1d.2xlarge (8 vCPU, 64 GiB)
"""

import os, sys, gc, re, time, unicodedata
import numpy as np
import pandas as pd
from collections import defaultdict
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
import warnings; warnings.filterwarnings("ignore")

os.environ.setdefault("OMP_NUM_THREADS", "8")

try:
    from rapidfuzz.distance import Levenshtein as _Lev
    def lev_ratio(a, b):
        if not a or not b: return 0.0
        return _Lev.normalized_similarity(a, b)
except ImportError:
    from difflib import SequenceMatcher
    def lev_ratio(a, b):
        if not a or not b: return 0.0
        return SequenceMatcher(None, a, b).ratio()

T0 = time.time()
def log(msg):
    e = time.time() - T0; m, s = divmod(int(e), 60)
    print(f"[{m:02d}:{s:02d}] {msg}", flush=True)

# =====================================================================
#  CONFIG
# =====================================================================
DATASET_ROOT = None
CAND_FILE = None
OUTPUT_DIR = None

def init_config():
    global DATASET_ROOT, CAND_FILE, OUTPUT_DIR
    
    # Auto-detect dataset root
    for p in [
        os.environ.get("DATASET_ROOT", ""),
        "/home/ubuntu/student_resource/student_resource",
        os.path.expanduser("~/student_resource/student_resource"),
        os.path.expanduser("~/student_resource"),
        r"C:\Users\sivar\OneDrive\Desktop\student_resource",
    ]:
        if p and os.path.isfile(os.path.join(p, "dataset", "train", "train_source1.tsv")):
            DATASET_ROOT = p; break
    if not DATASET_ROOT:
        raise FileNotFoundError("Cannot find dataset root (dataset/train/train_source1.tsv)")

    # Auto-detect candidate_pairs.tsv
    for p in [
        os.environ.get("CAND_FILE", ""),
        os.path.join(os.getcwd(), "output", "candidate_pairs.tsv"),
        os.path.join(os.getcwd(), "candidate_pairs.tsv"),
        os.path.expanduser("~/output/candidate_pairs.tsv"),
        os.path.expanduser("~/candidate_pairs.tsv"),
        "/home/ubuntu/output/candidate_pairs.tsv",
        "/home/ubuntu/candidate_pairs.tsv",
        r"C:\Users\sivar\Downloads\candidate_pairs.tsv",
        r"C:\Users\sivar\Downloads\improved_output\candidate_pairs.tsv",
    ]:
        if p and os.path.isfile(p):
            CAND_FILE = p; break

    if not CAND_FILE:
        for root, dirs, files in os.walk(os.path.expanduser("~")):
            if "candidate_pairs.tsv" in files:
                CAND_FILE = os.path.join(root, "candidate_pairs.tsv")
                break

    if not CAND_FILE:
        raise FileNotFoundError("Cannot find candidate_pairs.tsv")

    OUTPUT_DIR = os.path.join(os.getcwd(), "output_v2")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log(f"DATASET_ROOT: {DATASET_ROOT}")
    log(f"CANDIDATES:   {CAND_FILE}")
    log(f"OUTPUT:       {OUTPUT_DIR}")

# =====================================================================
#  TEXT CANONICALIZATION
# =====================================================================
_LEGAL = re.compile(
    r'\b('
    # English
    r'corporation|corp|incorporated|inc|limited|ltd|llc|l\.l\.c|llp|'
    r'co|company|group|holdings?|international|intl|'
    # India
    r'pvt\s*ltd|private\s*limited|pvt|private|enterprises?|'
    r'solutions?|services?|technologies|tech|traders?|stores?|'
    r'industries|associates|consultants?|'
    # France
    r'societe|ste|sarl|s\.a\.r\.l|sas|s\.a\.s|sa|s\.a|eurl|sci|snc|'
    # Other
    r'gmbh|ag|pty|plc|nv|bv|ab|oy|as'
    r')\b', re.IGNORECASE)

_ADDR_ABBREV = [
    (re.compile(r'\bst\b'),   'street'),
    (re.compile(r'\brd\b'),   'road'),
    (re.compile(r'\bave\b'),  'avenue'),
    (re.compile(r'\bblvd\b'), 'boulevard'),
    (re.compile(r'\bdr\b'),   'drive'),
    (re.compile(r'\bln\b'),   'lane'),
    (re.compile(r'\bct\b'),   'court'),
    (re.compile(r'\bopp\b'),  'opposite'),
    (re.compile(r'\bnr\b'),   'near'),
    (re.compile(r'\bdist\b'), 'district'),
]

def _base_norm(text):
    if pd.isna(text) or not text: return ""
    t = str(text).lower()
    t = unicodedata.normalize("NFKD", t)
    t = re.sub(r'[&]', ' and ', t)
    t = re.sub(r'[^\w\s]', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()

def canon_name(text):
    t = _base_norm(text)
    t = _LEGAL.sub(' ', t)
    return re.sub(r'\s+', ' ', t).strip()

def canon_addr(text):
    t = _base_norm(text)
    for pat, repl in _ADDR_ABBREV:
        t = pat.sub(repl, t)
    return re.sub(r'\s+', ' ', t).strip()

# =====================================================================
#  FEATURE ENGINEERING
# =====================================================================
def _jac(a, b):
    if not a or not b: return 0.0
    sa, sb = set(a.split()), set(b.split())
    u = len(sa | sb)
    return len(sa & sb) / u if u else 0.0

def _sorted_tokens(s):
    return " ".join(sorted(s.split())) if s else ""

def _extract_nums(s):
    if not s: return set()
    return set(re.findall(r'\b\d+\b', s))

def compute_features(s1n, s1a, s1cn, s1ca, s1co,
                     gn, ga, gcn, gca, gco, g_src):
    """Compute 18 features for each (s1, gallery) pair."""
    n = len(s1n)
    f = {}

    # -- Levenshtein variants --
    log(f"  feat: lev_name ({n:,} pairs)")
    f['lev_name']    = np.array([lev_ratio(a,b) for a,b in zip(s1n, gn)], np.float32)
    log(f"  feat: lev_cname")
    f['lev_cname']   = np.array([lev_ratio(a,b) for a,b in zip(s1cn, gcn)], np.float32)
    log(f"  feat: lev_sorted")
    s1s = [_sorted_tokens(x) for x in s1cn]
    gs  = [_sorted_tokens(x) for x in gcn]
    f['lev_sorted']  = np.array([lev_ratio(a,b) for a,b in zip(s1s, gs)], np.float32)
    log(f"  feat: lev_addr")
    f['lev_addr']    = np.array([lev_ratio(a,b) for a,b in zip(s1ca, gca)], np.float32)

    # -- Token Jaccard --
    f['jac_cname']   = np.array([_jac(a,b) for a,b in zip(s1cn, gcn)], np.float32)
    f['jac_addr']    = np.array([_jac(a,b) for a,b in zip(s1ca, gca)], np.float32)

    # -- Exact matches --
    f['exact_cname'] = (np.array(s1cn) == np.array(gcn)).astype(np.int8)
    f['country_match'] = (np.array(s1co) == np.array(gco)).astype(np.int8)

    # -- Length / structural --
    l1 = np.array([len(x) for x in s1cn], np.float32)
    l2 = np.array([len(x) for x in gcn], np.float32)
    mx = np.maximum(l1, l2)
    f['name_len_ratio'] = np.where(mx > 0, np.minimum(l1, l2) / mx, 1.0).astype(np.float32)

    f['name_tok_diff'] = np.abs(
        np.array([len(x.split()) for x in s1cn]) -
        np.array([len(x.split()) for x in gcn])
    ).astype(np.int8)

    # -- Prefix matches --
    f['prefix3'] = np.array([
        1 if len(a)>=3 and len(b)>=3 and a[:3]==b[:3] else 0
        for a,b in zip(s1cn, gcn)], np.int8)
    f['prefix5'] = np.array([
        1 if len(a)>=5 and len(b)>=5 and a[:5]==b[:5] else 0
        for a,b in zip(s1cn, gcn)], np.int8)

    # -- Common address tokens --
    f['common_addr_tok'] = np.array([
        len(set(a.split()) & set(b.split())) if a and b else 0
        for a,b in zip(s1ca, gca)], np.int16)

    # -- Digit / postal code features --
    log(f"  feat: digit_jaccard + zip_conflict")
    dj_vals = np.empty(n, np.float32)
    zc_vals = np.zeros(n, np.int8)
    for i in range(n):
        n1 = _extract_nums(s1ca[i])
        n2 = _extract_nums(gca[i])
        if n1 and n2:
            inter = len(n1 & n2)
            union = len(n1 | n2)
            dj_vals[i] = inter / union if union else 0.0
            z1 = {x for x in n1 if len(x) in (5,6)}
            z2 = {x for x in n2 if len(x) in (5,6)}
            if z1 and z2 and not (z1 & z2):
                zc_vals[i] = 1
        else:
            dj_vals[i] = 0.0
    f['digit_jaccard'] = dj_vals
    f['zip_conflict']  = zc_vals

    # -- Source type --
    f['is_s2'] = np.array([1 if s.startswith("S2") else 0 for s in g_src], np.int8)

    # -- Character-level overlap (fast approximation) --
    f['char_jac'] = np.array([
        len(set(a) & set(b)) / max(len(set(a) | set(b)), 1)
        if a and b else 0.0
        for a,b in zip(s1cn, gcn)], np.float32)

    return pd.DataFrame(f)

# =====================================================================
#  MACRO F0.5
# =====================================================================
def macro_f05(pred_dict, gt_dict, all_s1):
    scores = []
    for s1 in all_s1:
        pred = pred_dict.get(s1, set())
        true = gt_dict.get(s1, set())
        if not true and not pred:   scores.append(1.0)
        elif not true:              scores.append(0.0)
        elif not pred:              scores.append(0.0)
        else:
            tp = len(pred & true)
            p = tp / len(pred); r = tp / len(true)
            scores.append((1.25*p*r)/(0.25*p+r) if p+r else 0.0)
    return float(np.mean(scores))

# =====================================================================
#  LOAD DATA
# =====================================================================
def load_tsv(path):
    return pd.read_csv(path, sep="\t", dtype=str)

def build_lookup(df):
    """Build entity_id -> (norm_name, norm_addr, canon_name, canon_addr, country) dict."""
    lookup = {}
    for _, r in df.iterrows():
        eid = str(r["entity_id"]).strip()
        nn = _base_norm(r.get("business_name",""))
        na = _base_norm(r.get("business_address",""))
        cn = canon_name(r.get("business_name",""))
        ca = canon_addr(r.get("business_address",""))
        co = str(r.get("country","")).strip() if pd.notna(r.get("country","")) else ""
        lookup[eid] = (nn, na, cn, ca, co)
    return lookup

def build_lookup_fast(df):
    """Vectorized lookup builder — much faster for millions of rows."""
    log(f"  Building lookup for {len(df):,} entities ...")
    eids = df["entity_id"].astype(str).str.strip().values
    nns  = df["business_name"].apply(_base_norm).values
    nas  = df["business_address"].apply(_base_norm).values
    cns  = df["business_name"].apply(canon_name).values
    cas  = df["business_address"].apply(canon_addr).values
    cos  = df["country"].fillna("").astype(str).str.strip().values
    return dict(zip(eids, zip(nns, nas, cns, cas, cos)))

# =====================================================================
#  MAIN
# =====================================================================
def main():
    init_config()
    td = os.path.join(DATASET_ROOT, "dataset", "train")
    tt = os.path.join(DATASET_ROOT, "dataset", "test")

    # ── Load training data ───────────────────────────────────
    log("=== STAGE 1: Load training data ===")
    tr_s1 = load_tsv(os.path.join(td, "train_source1.tsv"))
    tr_s2 = load_tsv(os.path.join(td, "train_source2.tsv"))
    tr_s3 = load_tsv(os.path.join(td, "train_source3.tsv"))
    gt_raw = load_tsv(os.path.join(td, "train_ground_truth.tsv"))
    log(f"  S1={len(tr_s1):,}  S2={len(tr_s2):,}  S3={len(tr_s3):,}  GT={len(gt_raw):,}")

    # Parse GT
    gt_dict = {}
    gt_pos_pairs = []
    for _, r in gt_raw.iterrows():
        s1 = str(r["source1_entity_id"]).strip()
        val = r.get("matched_entity_ids", "")
        if pd.isna(val) or str(val).strip() in ("", "nan"):
            gt_dict[s1] = set(); continue
        ids = [x.strip() for x in str(val).split(",") if x.strip() and x.strip()!="nan"]
        gt_dict[s1] = set(ids)
        for mid in ids:
            gt_pos_pairs.append((s1, mid))
    log(f"  Positive GT pairs: {len(gt_pos_pairs):,}")
    log(f"  S1 with matches: {sum(1 for v in gt_dict.values() if v):,}")
    log(f"  Singletons in GT: {sum(1 for v in gt_dict.values() if not v):,}")

    # Build lookups
    log("  Building entity lookups ...")
    gallery_tr = pd.concat([tr_s2, tr_s3], ignore_index=True)
    lk_s1_tr = build_lookup_fast(tr_s1)
    lk_g_tr  = build_lookup_fast(gallery_tr)
    del tr_s2, tr_s3, gallery_tr; gc.collect()

    # ── Create training examples ─────────────────────────────
    log("\n=== STAGE 2: Create training pairs ===")
    # Sample positive pairs (600k is the sweet spot for LightGBM convergence and speed)
    np.random.seed(42)
    pos = gt_pos_pairs
    if len(pos) > 600_000:
        idx = np.random.choice(len(pos), 600_000, replace=False)
        pos = [pos[i] for i in idx]
    log(f"  Positives: {len(pos):,}")

    # Country-aware negative sampling (3:1 ratio) - 100% Vectorized
    log("  Vectorized negative sampling ...")
    country_gallery = {}
    cg_lists = defaultdict(list)
    for eid, vals in lk_g_tr.items():
        cg_lists[vals[4]].append(eid)
    for co, lst in cg_lists.items():
        country_gallery[co] = np.array(lst)
    del cg_lists

    neg = []
    neg_per_pos = 3
    country_pos = defaultdict(list)
    for s1_id, _ in pos:
        if s1_id in lk_s1_tr:
            co = lk_s1_tr[s1_id][4]
            country_pos[co].append(s1_id)

    for co, s1_list in country_pos.items():
        pool = country_gallery.get(co)
        if pool is None or len(pool) == 0:
            continue
        n_req = len(s1_list) * neg_per_pos
        rand_idx = np.random.randint(0, len(pool), size=n_req)
        sampled_gids = pool[rand_idx]
        repeated_s1 = np.repeat(s1_list, neg_per_pos)
        neg.extend(zip(repeated_s1, sampled_gids))

    log(f"  Negatives: {len(neg):,}")

    # Combine
    all_pairs = [(s, g, 1) for s, g in pos] + [(s, g, 0) for s, g in neg]
    np.random.shuffle(all_pairs)
    tr_s1_ids = [p[0] for p in all_pairs]
    tr_g_ids  = [p[1] for p in all_pairs]
    tr_labels = np.array([p[2] for p in all_pairs], np.int8)
    log(f"  Total training: {len(all_pairs):,}  (pos rate: {tr_labels.mean():.3f})")

    # ── Compute training features ────────────────────────────
    log("\n=== STAGE 3: Training features ===")
    _empty = ("", "", "", "", "")
    s1n  = [lk_s1_tr.get(s, _empty)[0] for s in tr_s1_ids]
    s1a  = [lk_s1_tr.get(s, _empty)[1] for s in tr_s1_ids]
    s1cn = [lk_s1_tr.get(s, _empty)[2] for s in tr_s1_ids]
    s1ca = [lk_s1_tr.get(s, _empty)[3] for s in tr_s1_ids]
    s1co = [lk_s1_tr.get(s, _empty)[4] for s in tr_s1_ids]
    gn   = [lk_g_tr.get(g, _empty)[0] for g in tr_g_ids]
    ga   = [lk_g_tr.get(g, _empty)[1] for g in tr_g_ids]
    gcn  = [lk_g_tr.get(g, _empty)[2] for g in tr_g_ids]
    gca  = [lk_g_tr.get(g, _empty)[3] for g in tr_g_ids]
    gco  = [lk_g_tr.get(g, _empty)[4] for g in tr_g_ids]

    tr_feats = compute_features(s1n, s1a, s1cn, s1ca, s1co,
                                gn, ga, gcn, gca, gco, tr_g_ids)
    tr_feats["label"] = tr_labels
    feat_cols = [c for c in tr_feats.columns if c != "label"]
    log(f"  Feature cols: {feat_cols}")
    del s1n, s1a, s1cn, s1ca, s1co, gn, ga, gcn, gca, gco; gc.collect()

    # ── Train LightGBM ───────────────────────────────────────
    log("\n=== STAGE 4: Train LightGBM ===")
    params = dict(objective="binary", metric="binary_logloss",
                  learning_rate=0.05, num_leaves=63, max_depth=8,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
                  n_estimators=800, n_jobs=8, seed=42, verbose=-1)

    # 3-fold GroupKFold for threshold optimization
    gkf = GroupKFold(n_splits=3)
    tr_feats["prob"] = np.float32(0)
    models = []
    groups = np.array(tr_s1_ids)

    for fold, (ti, vi) in enumerate(gkf.split(tr_feats, groups=groups)):
        log(f"  Fold {fold+1}/3")
        Xtr = tr_feats.iloc[ti][feat_cols]; ytr = tr_feats.iloc[ti]["label"]
        Xvl = tr_feats.iloc[vi][feat_cols]; yvl = tr_feats.iloc[vi]["label"]
        dt = lgb.Dataset(Xtr, label=ytr)
        dv = lgb.Dataset(Xvl, label=yvl, reference=dt)
        m = lgb.train(params, dt, valid_sets=[dt, dv],
                      callbacks=[lgb.early_stopping(30), lgb.log_evaluation(200)])
        models.append(m)
        tr_feats.iloc[vi, tr_feats.columns.get_loc("prob")] = \
            m.predict(Xvl, num_iteration=m.best_iteration).astype(np.float32)
        log(f"    best_iter={m.best_iteration}")

    # Threshold sweep on OOF predictions
    log("\n  Threshold sweep on OOF ...")
    oof_s1 = np.array(tr_s1_ids)
    oof_g  = np.array(tr_g_ids)
    oof_p  = tr_feats["prob"].values
    all_tr_s1 = set(tr_s1["entity_id"].astype(str))

    best_t, best_f = 0.5, 0.0
    for t in np.arange(0.30, 0.96, 0.01):
        mask = oof_p > t
        pred = defaultdict(set)
        for s, g in zip(oof_s1[mask], oof_g[mask]):
            pred[s].add(g)
        f = macro_f05(pred, gt_dict, all_tr_s1)
        if f > best_f:
            best_f, best_t = f, t
    log(f"  BEST THRESHOLD: {best_t:.2f}  F0.5={best_f:.4f}")

    # Feature importance
    imp = models[0].feature_importance(importance_type="gain")
    for fn, fi in sorted(zip(feat_cols, imp), key=lambda x: -x[1])[:10]:
        log(f"    {fn}: {fi:.0f}")

    del tr_feats, lk_s1_tr, lk_g_tr; gc.collect()

    # ── Load test data ───────────────────────────────────────
    log("\n=== STAGE 5: Load test data ===")
    te_s1 = load_tsv(os.path.join(tt, "test_source1.tsv"))
    te_s2 = load_tsv(os.path.join(tt, "test_source2.tsv"))
    te_s3 = load_tsv(os.path.join(tt, "test_source3.tsv"))
    log(f"  S1={len(te_s1):,}  S2={len(te_s2):,}  S3={len(te_s3):,}")

    gallery_te = pd.concat([te_s2, te_s3], ignore_index=True)
    lk_s1_te = build_lookup_fast(te_s1)
    lk_g_te  = build_lookup_fast(gallery_te)
    del te_s2, te_s3, gallery_te; gc.collect()

    # ── Load & explode candidate_pairs.tsv ───────────────────
    log("\n=== STAGE 6: Load candidates ===")
    cp = pd.read_csv(CAND_FILE, sep="\t", dtype=str)
    cp["candidate_entity_ids"] = cp["candidate_entity_ids"].fillna("")
    log(f"  Candidate rows: {len(cp):,}")

    # Explode into individual pairs
    non_empty = cp[cp["candidate_entity_ids"] != ""].copy()
    non_empty["cid_list"] = non_empty["candidate_entity_ids"].str.split(",")
    pairs = non_empty.explode("cid_list")[["source1_entity_id", "cid_list"]].copy()
    pairs["cid_list"] = pairs["cid_list"].str.strip()
    pairs = pairs[pairs["cid_list"] != ""].reset_index(drop=True)
    pairs.columns = ["s1_id", "g_id"]
    log(f"  Exploded candidate pairs: {len(pairs):,}")
    del cp, non_empty; gc.collect()

    # ── Compute test features ────────────────────────────────
    log("\n=== STAGE 7: Test features ===")
    te_s1_arr = pairs["s1_id"].values
    te_g_arr  = pairs["g_id"].values

    s1n  = [lk_s1_te.get(s, _empty)[0] for s in te_s1_arr]
    s1a  = [lk_s1_te.get(s, _empty)[1] for s in te_s1_arr]
    s1cn = [lk_s1_te.get(s, _empty)[2] for s in te_s1_arr]
    s1ca = [lk_s1_te.get(s, _empty)[3] for s in te_s1_arr]
    s1co = [lk_s1_te.get(s, _empty)[4] for s in te_s1_arr]
    gn   = [lk_g_te.get(g, _empty)[0] for g in te_g_arr]
    ga   = [lk_g_te.get(g, _empty)[1] for g in te_g_arr]
    gcn  = [lk_g_te.get(g, _empty)[2] for g in te_g_arr]
    gca  = [lk_g_te.get(g, _empty)[3] for g in te_g_arr]
    gco  = [lk_g_te.get(g, _empty)[4] for g in te_g_arr]

    te_feats = compute_features(s1n, s1a, s1cn, s1ca, s1co,
                                gn, ga, gcn, gca, gco, te_g_arr)
    del s1n, s1a, s1cn, s1ca, s1co, gn, ga, gcn, gca, gco; gc.collect()

    # ── Predict ──────────────────────────────────────────────
    log("\n=== STAGE 8: Predict ===")
    te_feats["prob"] = np.float32(0)
    for m in models:
        te_feats["prob"] += m.predict(
            te_feats[feat_cols], num_iteration=m.best_iteration
        ).astype(np.float32) / len(models)

    # Save raw probabilities (prob >= 0.20) for offline threshold experiments
    p_mask = te_feats["prob"].values >= 0.20
    probs_path = os.path.join(OUTPUT_DIR, "test_probabilities_top.tsv")
    pd.DataFrame({
        "s1_id": te_s1_arr[p_mask],
        "g_id": te_g_arr[p_mask],
        "prob": te_feats["prob"].values[p_mask]
    }).to_csv(probs_path, sep="\t", index=False)
    log(f"  Saved top probabilities ({p_mask.sum():,} pairs): {probs_path}")

    # ── Greedy 1-to-1 assignment ─────────────────────────────
    log(f"\n=== STAGE 9: Greedy 1-to-1 (thr={best_t:.2f}) ===")
    probs = te_feats["prob"].values
    mask = probs >= best_t

    scored = pd.DataFrame({
        "s1_id": te_s1_arr[mask],
        "g_id":  te_g_arr[mask],
        "prob":  probs[mask]
    })
    log(f"  Pairs above threshold: {len(scored):,}")

    # Sort by prob desc, deduplicate targets (each g_id -> 1 s1_id)
    scored.sort_values("prob", ascending=False, inplace=True)
    deduped = scored.drop_duplicates(subset=["g_id"], keep="first")
    log(f"  After 1-to-1 dedup: {len(deduped):,}  (removed {len(scored)-len(deduped):,})")

    # Group by S1
    matches = deduped.groupby("s1_id")["g_id"].apply(lambda s: sorted(set(s))).to_dict()

    # ── Write output ─────────────────────────────────────────
    log("\n=== STAGE 10: Write output ===")
    all_test_s1 = te_s1["entity_id"].astype(str).values

    # matching_results.tsv
    rows_m = [{"source1_entity_id": sid,
               "matched_entity_ids": ",".join(matches.get(sid, []))}
              for sid in all_test_s1]
    mp = os.path.join(OUTPUT_DIR, "matching_results.tsv")
    pd.DataFrame(rows_m).to_csv(mp, sep="\t", index=False)
    n_match = sum(1 for r in rows_m if r["matched_entity_ids"])
    n_sing  = sum(1 for r in rows_m if not r["matched_entity_ids"])
    log(f"  matching_results.tsv: {len(rows_m):,} rows ({n_match:,} matched, {n_sing:,} singletons)")

    # candidate_pairs.tsv (copy original)
    import shutil
    cp_dst = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
    shutil.copy2(CAND_FILE, cp_dst)
    log(f"  candidate_pairs.tsv: copied")

    # ── Validate ─────────────────────────────────────────────
    validator = os.path.join(DATASET_ROOT, "utils", "validate_submission.py")
    if os.path.isfile(validator):
        cmd = (f"{sys.executable} {validator} "
               f"--matching {mp} --candidate {cp_dst} "
               f"--test-dir {os.path.join(DATASET_ROOT, 'dataset', 'test')}")
        log(f"  Validator: {cmd}")
        os.system(cmd)

    log(f"\n{'='*60}")
    log(f"RESCORER v2 COMPLETE  ({time.time()-T0:.0f}s)")
    log(f"Threshold: {best_t:.2f}   Matched: {n_match:,}   Singletons: {n_sing:,}")
    log(f"Output: {OUTPUT_DIR}")
    log(f"{'='*60}")


if __name__ == "__main__":
    main()
