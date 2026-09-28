#!/usr/bin/env python3
"""Amazon ML Challenge 2026 — Entity Matching Pipeline (Lexical)
Architecture: Multi-route TF-IDF blocking → Features → LightGBM → F0.5 threshold.
Memory-safe: no .toarray(), chunked candidate I/O, parquet caching."""

import os, sys, gc, re, time, json, hashlib, unicodedata
import numpy as np
import pandas as pd
from typing import Dict, Set, Tuple, Optional
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

# ── String distance ──────────────────────────────────────────
try:
    from rapidfuzz.distance import Levenshtein as _rl
    def lev_ratio(a: str, b: str) -> float:
        if not a or not b: return 0.0
        return _rl.normalized_similarity(a, b)
except ImportError:
    try:
        import Levenshtein as _pl
        def lev_ratio(a, b):
            if not a or not b: return 0.0
            return _pl.ratio(a, b)
    except ImportError:
        from difflib import SequenceMatcher
        def lev_ratio(a, b):
            if not a or not b: return 0.0
            return SequenceMatcher(None, a, b).ratio()

# ── Memory tracking ──────────────────────────────────────────
try:
    import psutil
    def _mem_mb(): return psutil.Process(os.getpid()).memory_info().rss / (1024**2)
except ImportError:
    def _mem_mb(): return -1.0

def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}|{_mem_mb():,.0f}MB] {msg}", flush=True)

# ══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════
class Config:
    # ── Execution mode (override with env PIPELINE_MODE) ──
    #   smoke       → 1K S1 + 10K gallery   (< 2 min, validates full pipeline)
    #   validation  → 10 % S1 + full gallery (local CV)
    #   full        → everything
    MODE = os.environ.get("PIPELINE_MODE", "full")

    SMOKE_S1      = 1000
    SMOKE_GALLERY = 10000
    VAL_S1_FRAC   = 0.10

    # ── Output / cache ──
    OUT_DIR   = "/kaggle/working/output"
    CACHE_DIR = "/kaggle/working/cache"

    # ── Paths (set by init_paths) ──
    DATASET_ROOT = ""
    TRAIN_S1 = TRAIN_S2 = TRAIN_S3 = TRAIN_GT = ""
    TEST_S1  = TEST_S2  = TEST_S3  = VALIDATOR = ""

    @classmethod
    def init_paths(cls):
        root = None
        # 1. Kaggle auto-discovery
        if os.path.isdir("/kaggle/input"):
            for r, dirs, _ in os.walk("/kaggle/input"):
                if "student_resource" in dirs:
                    p = os.path.join(r, "student_resource")
                    if os.path.isfile(os.path.join(p, "dataset", "train", "train_source1.tsv")):
                        root = p; break
            if not root:
                root = "/kaggle/input/datasets/srkreddy6115/amazon-ml-challenge-2026/student_resource"
        # 2. Env var fallback (local testing)
        if not root:
            env = os.environ.get("DATASET_ROOT", "")
            if env and os.path.isdir(env): root = env
        if not root or not os.path.isdir(root):
            raise FileNotFoundError(f"Cannot find dataset root. Got: {root}")

        cls.DATASET_ROOT = root
        td = os.path.join(root, "dataset", "train")
        tt = os.path.join(root, "dataset", "test")
        cls.TRAIN_S1 = os.path.join(td, "train_source1.tsv")
        cls.TRAIN_S2 = os.path.join(td, "train_source2.tsv")
        cls.TRAIN_S3 = os.path.join(td, "train_source3.tsv")
        cls.TRAIN_GT = os.path.join(td, "train_ground_truth.tsv")
        cls.TEST_S1  = os.path.join(tt, "test_source1.tsv")
        cls.TEST_S2  = os.path.join(tt, "test_source2.tsv")
        cls.TEST_S3  = os.path.join(tt, "test_source3.tsv")
        cls.VALIDATOR = os.path.join(root, "utils", "validate_submission.py")

    # ── Blocking ──
    K_NAME          = 30
    K_ADDR          = 10
    BLOCK_THRESH    = 0.20
    CHUNK_SIZE      = 500         # Safe chunk size for Kaggle/Colab
    COUNTRY_AWARE   = True

    # ── LightGBM ──
    LGBM = dict(objective="binary", metric="binary_logloss",
                learning_rate=0.05, num_leaves=63, max_depth=8,
                feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
                n_estimators=1000, n_jobs=-1, seed=42, verbose=-1)
    N_FOLDS        = 3
    EARLY_STOP     = 30
    THRESHOLD      = 0.55

    @classmethod
    def cache_hash(cls) -> str:
        d = dict(K=cls.K_NAME, KA=cls.K_ADDR, T=cls.BLOCK_THRESH, C=cls.COUNTRY_AWARE)
        return hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()[:8]

# ══════════════════════════════════════════════════════════════
#  STARTUP
# ══════════════════════════════════════════════════════════════
def startup_report():
    import platform, multiprocessing as mp
    log(f"MODE={Config.MODE}  COUNTRY_AWARE={Config.COUNTRY_AWARE}")
    log(f"DATASET_ROOT: {Config.DATASET_ROOT}")
    log(f"K_NAME={Config.K_NAME}  K_ADDR={Config.K_ADDR}  CHUNK={Config.CHUNK_SIZE}")
    log(f"Python {sys.version.split()[0]}  CPUs={mp.cpu_count()}  OS={platform.system()}")
    try:
        import torch
        log(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
    except Exception:
        log("GPU: torch unavailable")
    # Validate paths
    checks = {"TRAIN_S1": Config.TRAIN_S1, "TRAIN_S2": Config.TRAIN_S2,
              "TRAIN_S3": Config.TRAIN_S3, "TRAIN_GT": Config.TRAIN_GT,
              "TEST_S1": Config.TEST_S1, "TEST_S2": Config.TEST_S2,
              "TEST_S3": Config.TEST_S3, "VALIDATOR": Config.VALIDATOR}
    missing = []
    for label, path in checks.items():
        ok = os.path.isfile(path)
        log(f"  {label}: {path} [{'OK' if ok else 'MISSING'}]")
        if not ok: missing.append(path)
    if missing:
        raise FileNotFoundError("Missing:\n  " + "\n  ".join(missing))

# ══════════════════════════════════════════════════════════════
#  TEXT NORMALISATION
# ══════════════════════════════════════════════════════════════
def normalize(text) -> str:
    if pd.isna(text): return ""
    t = str(text).lower()
    t = unicodedata.normalize("NFKD", t)
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def has_non_latin(text) -> bool:
    if pd.isna(text): return False
    return any(ord(c) > 255 for c in str(text))

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalised columns; preserve originals."""
    df = df.copy()
    df["norm_name"] = df["business_name"].apply(normalize)
    df["norm_addr"] = df["business_address"].apply(normalize)
    df["is_nonlatin"] = df["business_name"].apply(has_non_latin).astype(np.int8)
    return df

# ══════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════
def load_ground_truth() -> Tuple[pd.DataFrame, Dict[str, Set[str]]]:
    raw = pd.read_csv(Config.TRAIN_GT, sep="\t", dtype=str)
    log(f"Loaded GT: {len(raw):,} rows")
    gt_dict: Dict[str, Set[str]] = {}
    rows = []
    for _, r in raw.iterrows():
        s1 = str(r["source1_entity_id"]).strip()
        val = r.get("matched_entity_ids", "")
        if pd.isna(val) or str(val).strip() in ("", "nan"):
            gt_dict[s1] = set()
            continue
        ids = [x.strip() for x in str(val).split(",") if x.strip() and x.strip() != "nan"]
        gt_dict[s1] = set(ids)
        for mid in ids:
            rows.append({"source1_entity_id": s1, "matched_id": mid})
    gt_pairs = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["source1_entity_id","matched_id"])
    log(f"  Positive pairs: {len(gt_pairs):,}  S1 with GT: {len(gt_dict):,}")
    return gt_pairs, gt_dict

def load_sources(split: str, gt_dict: Optional[Dict]=None):
    paths = {"train": (Config.TRAIN_S1, Config.TRAIN_S2, Config.TRAIN_S3),
             "test":  (Config.TEST_S1,  Config.TEST_S2,  Config.TEST_S3)}
    s1 = pd.read_csv(paths[split][0], sep="\t", dtype=str)
    s2 = pd.read_csv(paths[split][1], sep="\t", dtype=str)
    s3 = pd.read_csv(paths[split][2], sep="\t", dtype=str)
    log(f"Loaded {split}: S1={len(s1):,}  S2={len(s2):,}  S3={len(s3):,}")

    mode = Config.MODE
    if mode == "smoke":
        # Pick S1 entities with GT matches to ensure positive training signal
        if gt_dict and split == "train":
            s1_with = [x for x in s1["entity_id"] if x in gt_dict and len(gt_dict[x])>0]
            s1_without = [x for x in s1["entity_id"] if x not in gt_dict or len(gt_dict.get(x,set()))==0]
            keep = s1_with[:Config.SMOKE_S1//2] + s1_without[:Config.SMOKE_S1//2]
            keep = keep[:Config.SMOKE_S1]
            s1 = s1[s1["entity_id"].isin(set(keep))].reset_index(drop=True)
            # Ensure gallery includes GT match targets
            needed = set()
            for sid in keep:
                needed.update(gt_dict.get(sid, set()))
            s2_gt = s2[s2["entity_id"].isin(needed)]
            s3_gt = s3[s3["entity_id"].isin(needed)]
            s2 = pd.concat([s2_gt, s2.head(Config.SMOKE_GALLERY)]).drop_duplicates("entity_id").reset_index(drop=True)
            s3 = pd.concat([s3_gt, s3.head(Config.SMOKE_GALLERY)]).drop_duplicates("entity_id").reset_index(drop=True)
        else:
            s1 = s1.head(Config.SMOKE_S1)
            s2 = s2.head(Config.SMOKE_GALLERY)
            s3 = s3.head(Config.SMOKE_GALLERY)
    elif mode == "validation" and split == "train":
        s1 = s1.sample(frac=Config.VAL_S1_FRAC, random_state=42).reset_index(drop=True)

    log(f"  After mode={mode}: S1={len(s1):,}  S2={len(s2):,}  S3={len(s3):,}")
    return s1, s2, s3

# ══════════════════════════════════════════════════════════════
#  MEMORY-SAFE SPARSE TOP-K BLOCKING
# ══════════════════════════════════════════════════════════════
def sparse_topk(Q: sp.csr_matrix, G_T: sp.csc_matrix, k: int,
                threshold: float, chunk: int,
                q_ids: np.ndarray, g_ids: np.ndarray,
                route: str, cache_dir: str) -> pd.DataFrame:
    """Retrieve top-k gallery matches per query using sparse CSR direct access.
    Never materialises a dense similarity vector. Flushes to parquet periodically."""
    n = Q.shape[0]
    FLUSH = 300_000
    buf_q, buf_g, buf_v = [], [], []
    pq_files = []
    total = 0

    def _flush():
        nonlocal total
        if not buf_q: return
        df = pd.DataFrame({"s1_id": buf_q, "matched_id": buf_g,
                           f"{route}_score": np.array(buf_v, dtype=np.float32)})
        p = os.path.join(cache_dir, f"blk_{route}_{len(pq_files):05d}.parquet")
        df.to_parquet(p, engine="pyarrow", index=False)
        pq_files.append(p)
        total += len(df)
        buf_q.clear(); buf_g.clear(); buf_v.clear()
        del df

    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        sim = (Q[start:end] @ G_T).tocsr()

        for i in range(sim.shape[0]):
            p0, p1 = sim.indptr[i], sim.indptr[i+1]
            if p0 == p1: continue
            cols = sim.indices[p0:p1]
            vals = sim.data[p0:p1]
            mask = vals >= threshold
            cols, vals = cols[mask], vals[mask]
            if len(cols) == 0: continue
            actual_k = min(k, len(cols))
            if actual_k < len(cols):
                tk = np.argpartition(vals, -actual_k)[-actual_k:]
                cols, vals = cols[tk], vals[tk]

            qi = start + i
            buf_q.extend([q_ids[qi]] * len(cols))
            buf_g.extend(g_ids[cols].tolist())
            buf_v.extend(vals.tolist())

        del sim
        if len(buf_q) >= FLUSH:
            _flush(); gc.collect()
        if start > 0 and start % (chunk * 200) == 0:
            log(f"    {route}: {end:,}/{n:,}  pairs={total+len(buf_q):,}")

    _flush()
    if pq_files:
        out = pd.concat([pd.read_parquet(f) for f in pq_files], ignore_index=True)
        for f in pq_files: os.remove(f)
    else:
        out = pd.DataFrame(columns=["s1_id", "matched_id", f"{route}_score"])
    out["s1_id"] = out["s1_id"].astype("category")
    out["matched_id"] = out["matched_id"].astype("category")
    log(f"    {route}: {len(out):,} total candidates")
    return out

class Blocker:
    def __init__(self):
        self.vn = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,4),
                                  min_df=5, max_df=0.5, max_features=40000, dtype=np.float32)
        self.va = TfidfVectorizer(analyzer="word", ngram_range=(1,2),
                                  min_df=5, max_df=0.5, max_features=40000, dtype=np.float32)

    def fit(self, gallery: pd.DataFrame):
        log("Fitting TF-IDF on gallery...")
        self.g_ids = gallery["entity_id"].values
        self.g_countries = gallery["country"].fillna("").values
        self.Gn = self.vn.fit_transform(gallery["norm_name"].fillna(""))
        self.Ga = self.va.fit_transform(gallery["norm_addr"].fillna(""))
        log(f"  Name TF-IDF: {self.Gn.shape}  nnz={self.Gn.nnz:,}")
        log(f"  Addr TF-IDF: {self.Ga.shape}  nnz={self.Ga.nnz:,}")
        if Config.COUNTRY_AWARE:
            self.cidx = {}
            for c in np.unique(self.g_countries):
                if c: self.cidx[c] = np.where(self.g_countries == c)[0]
            log(f"  Country partitions: {len(self.cidx)}")

    def generate(self, s1: pd.DataFrame, cache_dir: str) -> pd.DataFrame:
        Sn = self.vn.transform(s1["norm_name"].fillna(""))
        Sa = self.va.transform(s1["norm_addr"].fillna(""))
        s1_ids = s1["entity_id"].values
        s1_countries = s1["country"].fillna("").values
        k_n, k_a = Config.K_NAME, Config.K_ADDR
        thr, ch = Config.BLOCK_THRESH, Config.CHUNK_SIZE

        if Config.COUNTRY_AWARE and hasattr(self, "cidx") and len(self.cidx) > 0:
            return self._gen_country(Sn, Sa, s1_ids, s1_countries, k_n, k_a, thr, ch, cache_dir)
        return self._gen_global(Sn, Sa, s1_ids, k_n, k_a, thr, ch, cache_dir)

    def _gen_global(self, Sn, Sa, s1_ids, k_n, k_a, thr, ch, cd):
        log("Global blocking...")
        GnT = self.Gn.T.tocsc()
        GaT = self.Ga.T.tocsc()
        nc = sparse_topk(Sn, GnT, k_n, thr, ch, s1_ids, self.g_ids, "name", cd)
        del GnT; gc.collect()
        ac = sparse_topk(Sa, GaT, k_a, thr, ch, s1_ids, self.g_ids, "addr", cd)
        del GaT; gc.collect()
        return self._merge(nc, ac)

    def _gen_country(self, Sn, Sa, s1_ids, s1_countries, k_n, k_a, thr, ch, cd):
        log("Country-aware blocking...")
        all_nc, all_ac = [], []
        processed_s1 = set()
        for country, g_idx in self.cidx.items():
            s1_mask = s1_countries == country
            if not np.any(s1_mask): continue
            s1_idx = np.where(s1_mask)[0]
            processed_s1.update(s1_idx.tolist())
            log(f"  [{country}] S1={len(s1_idx):,} × Gallery={len(g_idx):,}")
            GnT = self.Gn[g_idx].T.tocsc()
            GaT = self.Ga[g_idx].T.tocsc()
            g_ids_c = self.g_ids[g_idx]
            nc = sparse_topk(Sn[s1_idx], GnT, k_n, thr, ch, s1_ids[s1_idx], g_ids_c, "name", cd)
            del GnT; gc.collect()
            ac = sparse_topk(Sa[s1_idx], GaT, k_a, thr, ch, s1_ids[s1_idx], g_ids_c, "addr", cd)
            del GaT; gc.collect()
            all_nc.append(nc); all_ac.append(ac)

        # Fallback: S1 entities with unknown/missing country → full gallery
        remaining = np.array([i for i in range(len(s1_ids)) if i not in processed_s1])
        if len(remaining) > 0:
            log(f"  [FALLBACK] S1={len(remaining):,} × Gallery={len(self.g_ids):,}")
            GnT = self.Gn.T.tocsc()
            GaT = self.Ga.T.tocsc()
            nc = sparse_topk(Sn[remaining], GnT, k_n, thr, ch, s1_ids[remaining], self.g_ids, "name", cd)
            del GnT; gc.collect()
            ac = sparse_topk(Sa[remaining], GaT, k_a, thr, ch, s1_ids[remaining], self.g_ids, "addr", cd)
            del GaT; gc.collect()
            all_nc.append(nc); all_ac.append(ac)

        nc = pd.concat(all_nc, ignore_index=True) if all_nc else pd.DataFrame(columns=["s1_id","matched_id","name_score"])
        ac = pd.concat(all_ac, ignore_index=True) if all_ac else pd.DataFrame(columns=["s1_id","matched_id","addr_score"])
        return self._merge(nc, ac)

    @staticmethod
    def _merge(nc: pd.DataFrame, ac: pd.DataFrame) -> pd.DataFrame:
        if nc.empty and ac.empty:
            return pd.DataFrame(columns=["s1_id","matched_id","name_score","addr_score","source"])
        # Ensure string types for merge
        for df in [nc, ac]:
            df["s1_id"] = df["s1_id"].astype(str)
            df["matched_id"] = df["matched_id"].astype(str)
        m = pd.merge(nc, ac, on=["s1_id","matched_id"], how="outer")
        m["name_score"] = m["name_score"].fillna(0).astype(np.float32)
        m["addr_score"] = m["addr_score"].fillna(0).astype(np.float32)
        m["source"] = np.where(m["matched_id"].str.startswith("S2"), "S2", "S3")
        m["s1_id"] = m["s1_id"].astype("category")
        m["matched_id"] = m["matched_id"].astype("category")
        log(f"  Merged candidates: {len(m):,}  (name={len(nc):,}, addr={len(ac):,})")
        return m

# ══════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════
def generate_features(cands: pd.DataFrame, s1: pd.DataFrame,
                      gallery: pd.DataFrame) -> pd.DataFrame:
    log(f"Generating features for {len(cands):,} pairs...")
    s1i = s1.set_index("entity_id")
    gi  = gallery.set_index("entity_id")

    c = cands.copy()
    c["s1_id"]      = c["s1_id"].astype(str)
    c["matched_id"] = c["matched_id"].astype(str)

    s1_nn = c["s1_id"].map(s1i["norm_name"]).fillna("")
    g_nn  = c["matched_id"].map(gi["norm_name"]).fillna("")
    s1_co = c["s1_id"].map(s1i["country"]).fillna("")
    g_co  = c["matched_id"].map(gi["country"]).fillna("")

    c["exact_name"]   = (s1_nn == g_nn).astype(np.int8)
    c["country_match"]= (s1_co == g_co).astype(np.int8)
    c["is_s2"]        = (c["source"] == "S2").astype(np.int8)
    c["is_s3"]        = (c["source"] == "S3").astype(np.int8)
    c["s1_name_len"]  = s1_nn.str.len().astype(np.int16)
    c["g_name_len"]   = g_nn.str.len().astype(np.int16)
    c["name_len_diff"]= np.abs(c["s1_name_len"] - c["g_name_len"]).astype(np.int16)

    log("  Computing Levenshtein ratios...")
    c["lev_name"] = np.array(
        [lev_ratio(a, b) for a, b in zip(s1_nn.values, g_nn.values)],
        dtype=np.float32)

    # Token Jaccard (cheap, useful)
    def _jaccard(a, b):
        if not a or not b: return 0.0
        sa, sb = set(a.split()), set(b.split())
        inter = len(sa & sb); union = len(sa | sb)
        return inter / union if union else 0.0
    c["token_jaccard"] = np.array(
        [_jaccard(a, b) for a, b in zip(s1_nn.values, g_nn.values)],
        dtype=np.float32)

    c.drop(columns=["source"], inplace=True, errors="ignore")
    log(f"  Features: {[x for x in c.columns if x not in ('s1_id','matched_id')]}")
    return c

# ══════════════════════════════════════════════════════════════
#  COMPETITION METRIC — per-S1 set-level macro F0.5
# ══════════════════════════════════════════════════════════════
def macro_f05(pred_dict: Dict[str, Set[str]],
              gt_dict: Dict[str, Set[str]],
              all_s1: Set[str]) -> float:
    """Exact competition metric: per-entity F0.5, macro-averaged over ALL S1."""
    scores = []
    for s1_id in all_s1:
        pred = pred_dict.get(s1_id, set())
        true = gt_dict.get(s1_id, set())
        if len(true) == 0 and len(pred) == 0:
            scores.append(1.0)
        elif len(true) == 0 and len(pred) > 0:
            scores.append(0.0)
        elif len(pred) == 0 and len(true) > 0:
            scores.append(0.0)
        else:
            tp = len(pred & true)
            prec = tp / len(pred)
            rec  = tp / len(true)
            if prec + rec == 0:
                scores.append(0.0)
            else:
                scores.append((1.25 * prec * rec) / (0.25 * prec + rec))
    return float(np.mean(scores))

def optimise_threshold(probs: np.ndarray, s1_ids: np.ndarray,
                       matched_ids: np.ndarray,
                       gt_dict: Dict[str, Set[str]],
                       all_val_s1: Set[str]) -> float:
    """Search threshold that maximises macro F0.5 on validation pairs."""
    log("Optimising threshold for macro F0.5...")
    best_t, best_f = 0.5, 0.0
    for t in np.arange(0.20, 0.92, 0.02):
        mask = probs > t
        pred: Dict[str, Set[str]] = {}
        for sid, mid in zip(s1_ids[mask], matched_ids[mask]):
            pred.setdefault(sid, set()).add(mid)
        f = macro_f05(pred, gt_dict, all_val_s1)
        if f > best_f:
            best_f, best_t = f, t
    log(f"  Best threshold={best_t:.2f}  Macro-F0.5={best_f:.4f}")
    return best_t

# ══════════════════════════════════════════════════════════════
#  OUTPUT
# ══════════════════════════════════════════════════════════════
def write_outputs(test_s1_ids, matches_dict: Dict[str, list],
                  cand_dict: Dict[str, list]):
    # matching_results.tsv
    rows_m = []
    for sid in test_s1_ids:
        mids = matches_dict.get(sid, [])
        rows_m.append({"source1_entity_id": sid,
                       "matched_entity_ids": ",".join(mids) if mids else ""})
    df_m = pd.DataFrame(rows_m)
    mp = os.path.join(Config.OUT_DIR, "matching_results.tsv")
    df_m.to_csv(mp, sep="\t", index=False)
    log(f"Wrote {mp}  ({len(df_m):,} rows)")

    # candidate_pairs.tsv
    rows_c = []
    for sid in test_s1_ids:
        cids = cand_dict.get(sid, [])
        rows_c.append({"source1_entity_id": sid,
                       "candidate_entity_ids": ",".join(cids) if cids else ""})
    df_c = pd.DataFrame(rows_c)
    cp = os.path.join(Config.OUT_DIR, "candidate_pairs.tsv")
    df_c.to_csv(cp, sep="\t", index=False)
    log(f"Wrote {cp}  ({len(df_c):,} rows)")

    # Run validator if available
    if os.path.isfile(Config.VALIDATOR):
        test_dir = os.path.join(Config.DATASET_ROOT, "dataset", "test")
        cmd = (f"{sys.executable} {Config.VALIDATOR} "
               f"--matching {mp} --candidate {cp} --test-dir {test_dir}")
        log(f"Running validator: {cmd}")
        os.system(cmd)

# ══════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════
def main():
    Config.init_paths()
    for d in (Config.OUT_DIR, Config.CACHE_DIR):
        os.makedirs(d, exist_ok=True)
    startup_report()

    # ── 1. Load GT ──
    gt_pairs, gt_dict = load_ground_truth()

    # ── 2. Load & preprocess train ──
    tr_s1, tr_s2, tr_s3 = load_sources("train", gt_dict)
    tr_s1 = preprocess(tr_s1)
    tr_s2 = preprocess(tr_s2)
    tr_s3 = preprocess(tr_s3)
    gallery_tr = pd.concat([tr_s2, tr_s3], ignore_index=True)
    log(f"Train gallery: {len(gallery_tr):,}")

    # ── 3. Blocking ──
    blocker = Blocker()
    blocker.fit(gallery_tr)
    cands = blocker.generate(tr_s1, Config.CACHE_DIR)
    log(f"Train candidates: {len(cands):,}")

    # ── 4. Label ──
    gt_set = set(zip(gt_pairs["source1_entity_id"], gt_pairs["matched_id"]))
    cands["label"] = np.array([
        1 if (str(s), str(m)) in gt_set else 0
        for s, m in zip(cands["s1_id"], cands["matched_id"])
    ], dtype=np.int8)
    pos = int(cands["label"].sum())
    log(f"  Positive: {pos:,}  Negative: {len(cands)-pos:,}")

    # ── 5. Features ──
    feats = generate_features(cands, tr_s1, gallery_tr)

    del tr_s2, tr_s3, gallery_tr, blocker
    gc.collect()

    # ── 6. Train LightGBM (GroupKFold by S1) ──
    feat_cols = [c for c in feats.columns if c not in ("s1_id","matched_id","label")]
    log(f"Feature columns: {feat_cols}")

    gkf = GroupKFold(n_splits=Config.N_FOLDS)
    feats["prob"] = np.float32(0)
    models = []
    s1_arr = feats["s1_id"].astype(str).values

    for fold, (ti, vi) in enumerate(gkf.split(feats, groups=s1_arr)):
        log(f"Fold {fold+1}/{Config.N_FOLDS}")
        Xtr = feats.iloc[ti][feat_cols]
        ytr = feats.iloc[ti]["label"]
        Xvl = feats.iloc[vi][feat_cols]
        yvl = feats.iloc[vi]["label"]
        dtrain = lgb.Dataset(Xtr, label=ytr)
        dval   = lgb.Dataset(Xvl, label=yvl, reference=dtrain)
        m = lgb.train(Config.LGBM, dtrain, valid_sets=[dtrain, dval],
                      callbacks=[lgb.early_stopping(Config.EARLY_STOP),
                                 lgb.log_evaluation(200)])
        models.append(m)
        feats.iloc[vi, feats.columns.get_loc("prob")] = m.predict(
            Xvl, num_iteration=m.best_iteration).astype(np.float32)
        log(f"  Fold {fold+1} best iter={m.best_iteration}")

    # ── 7. Optimise threshold ──
    all_train_s1 = set(tr_s1["entity_id"])
    Config.THRESHOLD = optimise_threshold(
        feats["prob"].values,
        feats["s1_id"].astype(str).values,
        feats["matched_id"].astype(str).values,
        gt_dict, all_train_s1)

    del feats, gt_pairs
    gc.collect()

    # ── 8. Test inference ──
    log("Loading test data...")
    te_s1, te_s2, te_s3 = load_sources("test")
    te_s1 = preprocess(te_s1)
    te_s2 = preprocess(te_s2)
    te_s3 = preprocess(te_s3)
    gallery_te = pd.concat([te_s2, te_s3], ignore_index=True)
    log(f"Test gallery: {len(gallery_te):,}")

    blocker_te = Blocker()
    blocker_te.fit(gallery_te)
    te_cands = blocker_te.generate(te_s1, Config.CACHE_DIR)
    log(f"Test candidates: {len(te_cands):,}")

    te_feats = generate_features(te_cands, te_s1, gallery_te)

    log("Predicting...")
    te_feats["prob"] = np.float32(0)
    for m in models:
        te_feats["prob"] += m.predict(
            te_feats[feat_cols], num_iteration=m.best_iteration).astype(np.float32) / len(models)

    # ── 9. Decode & write output ──
    mask = te_feats["prob"] > Config.THRESHOLD
    pos_df = te_feats.loc[mask, ["s1_id","matched_id"]]
    matches_dict: Dict[str, list] = {}
    for sid, mid in zip(pos_df["s1_id"].astype(str), pos_df["matched_id"].astype(str)):
        matches_dict.setdefault(sid, []).append(mid)

    cand_dict: Dict[str, list] = {}
    for sid, mid in zip(te_cands["s1_id"].astype(str), te_cands["matched_id"].astype(str)):
        cand_dict.setdefault(sid, []).append(mid)

    write_outputs(te_s1["entity_id"].values, matches_dict, cand_dict)
    log("Pipeline complete.")

if __name__ == "__main__":
    main()
