"""
Sparse Memory-Safe Candidate Blocking Engine
============================================
Performs multi-route sparse vector blocking across multiple source galleries.
Uses character n-grams (3-4 char_wb) for business names and word n-grams (1-2)
for addresses, coupled with country partitioning and chunked sparse dot products.
Never materializes full dense similarity matrices.
"""

import os
import gc
import logging
from typing import Dict, List, Optional, Tuple, Set
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import Config, default_config

logger = logging.getLogger(__name__)


def sparse_topk(
    Q: sp.csr_matrix,
    G_T: sp.csc_matrix,
    k: int,
    threshold: float,
    chunk_size: int,
    q_ids: np.ndarray,
    g_ids: np.ndarray,
    route: str,
    cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Computes top-k matches per query vector against gallery vectors using sparse dot product.
    Iterates in small chunks, extracting non-zero entries exceeding `threshold`.
    Flushes periodically to parquet if cache_dir is specified to keep memory bounded.
    """
    n_queries = Q.shape[0]
    flush_size = 300_000
    buf_q: List[str] = []
    buf_g: List[str] = []
    buf_v: List[float] = []
    parquet_files: List[str] = []
    total_pairs = 0

    def _flush_buffer():
        nonlocal total_pairs
        if not buf_q:
            return
        df_chunk = pd.DataFrame({
            "s1_id": buf_q,
            "matched_id": buf_g,
            f"{route}_score": np.array(buf_v, dtype=np.float32),
        })
        if cache_dir:
            fpath = os.path.join(cache_dir, f"blk_{route}_{len(parquet_files):05d}.parquet")
            df_chunk.to_parquet(fpath, engine="pyarrow", index=False)
            parquet_files.append(fpath)
        else:
            parquet_files.append(df_chunk)
        total_pairs += len(df_chunk)
        buf_q.clear()
        buf_g.clear()
        buf_v.clear()

    for start in range(0, n_queries, chunk_size):
        end = min(start + chunk_size, n_queries)
        sim_chunk = (Q[start:end] @ G_T).tocsr()

        for i in range(sim_chunk.shape[0]):
            p0, p1 = sim_chunk.indptr[i], sim_chunk.indptr[i + 1]
            if p0 == p1:
                continue
            cols = sim_chunk.indices[p0:p1]
            vals = sim_chunk.data[p0:p1]

            mask = vals >= threshold
            cols, vals = cols[mask], vals[mask]
            if len(cols) == 0:
                continue

            actual_k = min(k, len(cols))
            if actual_k < len(cols):
                top_idx = np.argpartition(vals, -actual_k)[-actual_k:]
                cols, vals = cols[top_idx], vals[top_idx]

            qi = start + i
            buf_q.extend([q_ids[qi]] * len(cols))
            buf_g.extend(g_ids[cols].tolist())
            buf_v.extend(vals.tolist())

        del sim_chunk
        if len(buf_q) >= flush_size:
            _flush_buffer()
            gc.collect()

    _flush_buffer()

    if not parquet_files:
        return pd.DataFrame(columns=["s1_id", "matched_id", f"{route}_score"])

    if cache_dir and isinstance(parquet_files[0], str):
        result = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
        for f in parquet_files:
            try:
                os.remove(f)
            except OSError:
                pass
    else:
        result = pd.concat(parquet_files, ignore_index=True)

    result["s1_id"] = result["s1_id"].astype("category")
    result["matched_id"] = result["matched_id"].astype("category")
    return result


class Blocker:
    """
    Multi-route blocking coordinator with dual TF-IDF indexation and country partitioning.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or default_config
        self.vec_name = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 4),
            min_df=5,
            max_df=0.5,
            max_features=self.config.max_features_char,
            dtype=np.float32,
        )
        self.vec_addr = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=5,
            max_df=0.5,
            max_features=self.config.max_features_word,
            dtype=np.float32,
        )
        self.g_ids: np.ndarray = np.array([])
        self.g_countries: np.ndarray = np.array([])
        self.Gn: Optional[sp.csr_matrix] = None
        self.Ga: Optional[sp.csr_matrix] = None
        self.country_indices: Dict[str, np.ndarray] = {}

    def fit(self, gallery_df: pd.DataFrame) -> "Blocker":
        """Fits TF-IDF vectorizers on normalized gallery entity records."""
        self.g_ids = gallery_df["entity_id"].values
        self.g_countries = gallery_df["country"].fillna("").astype(str).str.strip().values

        name_text = gallery_df["norm_name"].fillna("") if "norm_name" in gallery_df else gallery_df["business_name"].fillna("")
        addr_text = gallery_df["norm_addr"].fillna("") if "norm_addr" in gallery_df else gallery_df["business_address"].fillna("")

        self.Gn = self.vec_name.fit_transform(name_text)
        self.Ga = self.vec_addr.fit_transform(addr_text)

        if self.config.country_aware:
            self.country_indices = {}
            for c in np.unique(self.g_countries):
                if c:
                    self.country_indices[c] = np.where(self.g_countries == c)[0]

        return self

    def generate_candidates(self, query_df: pd.DataFrame, cache_dir: Optional[str] = None) -> pd.DataFrame:
        """Generates candidate pairs for query entities against the fitted gallery."""
        name_text = query_df["norm_name"].fillna("") if "norm_name" in query_df else query_df["business_name"].fillna("")
        addr_text = query_df["norm_addr"].fillna("") if "norm_addr" in query_df else query_df["business_address"].fillna("")

        Sn = self.vec_name.transform(name_text)
        Sa = self.vec_addr.transform(addr_text)
        q_ids = query_df["entity_id"].values
        q_countries = query_df["country"].fillna("").astype(str).str.strip().values

        if self.config.country_aware and self.country_indices:
            return self._generate_country_partitioned(Sn, Sa, q_ids, q_countries, cache_dir)
        return self._generate_global(Sn, Sa, q_ids, cache_dir)

    def _generate_global(
        self,
        Sn: sp.csr_matrix,
        Sa: sp.csr_matrix,
        q_ids: np.ndarray,
        cache_dir: Optional[str],
    ) -> pd.DataFrame:
        Gn_T = self.Gn.T.tocsc()
        Ga_T = self.Ga.T.tocsc()

        nc = sparse_topk(
            Sn, Gn_T, self.config.k_name, self.config.blocking_threshold,
            self.config.chunk_size, q_ids, self.g_ids, "name", cache_dir
        )
        del Gn_T
        gc.collect()

        ac = sparse_topk(
            Sa, Ga_T, self.config.k_addr, self.config.blocking_threshold,
            self.config.chunk_size, q_ids, self.g_ids, "addr", cache_dir
        )
        del Ga_T
        gc.collect()

        return self._merge_routes(nc, ac)

    def _generate_country_partitioned(
        self,
        Sn: sp.csr_matrix,
        Sa: sp.csr_matrix,
        q_ids: np.ndarray,
        q_countries: np.ndarray,
        cache_dir: Optional[str],
    ) -> pd.DataFrame:
        all_nc: List[pd.DataFrame] = []
        all_ac: List[pd.DataFrame] = []
        processed_query_indices: Set[int] = set()

        for country, g_idx in self.country_indices.items():
            q_mask = q_countries == country
            if not np.any(q_mask):
                continue
            sub_q_idx = np.where(q_mask)[0]
            processed_query_indices.update(sub_q_idx.tolist())

            Gn_c_T = self.Gn[g_idx].T.tocsc()
            Ga_c_T = self.Ga[g_idx].T.tocsc()
            g_ids_c = self.g_ids[g_idx]

            nc = sparse_topk(
                Sn[sub_q_idx], Gn_c_T, self.config.k_name, self.config.blocking_threshold,
                self.config.chunk_size, q_ids[sub_q_idx], g_ids_c, "name", cache_dir
            )
            del Gn_c_T
            gc.collect()

            ac = sparse_topk(
                Sa[sub_q_idx], Ga_c_T, self.config.k_addr, self.config.blocking_threshold,
                self.config.chunk_size, q_ids[sub_q_idx], g_ids_c, "addr", cache_dir
            )
            del Ga_c_T
            gc.collect()

            all_nc.append(nc)
            all_ac.append(ac)

        # Fallback for entities with missing/unmatched country
        unprocessed = np.array([i for i in range(len(q_ids)) if i not in processed_query_indices])
        if len(unprocessed) > 0:
            Gn_T = self.Gn.T.tocsc()
            Ga_T = self.Ga.T.tocsc()

            nc = sparse_topk(
                Sn[unprocessed], Gn_T, self.config.k_name, self.config.blocking_threshold,
                self.config.chunk_size, q_ids[unprocessed], self.g_ids, "name", cache_dir
            )
            del Gn_T
            gc.collect()

            ac = sparse_topk(
                Sa[unprocessed], Ga_T, self.config.k_addr, self.config.blocking_threshold,
                self.config.chunk_size, q_ids[unprocessed], self.g_ids, "addr", cache_dir
            )
            del Ga_T
            gc.collect()

            all_nc.append(nc)
            all_ac.append(ac)

        nc_combined = pd.concat(all_nc, ignore_index=True) if all_nc else pd.DataFrame(columns=["s1_id", "matched_id", "name_score"])
        ac_combined = pd.concat(all_ac, ignore_index=True) if all_ac else pd.DataFrame(columns=["s1_id", "matched_id", "addr_score"])

        return self._merge_routes(nc_combined, ac_combined)

    @staticmethod
    def _merge_routes(nc: pd.DataFrame, ac: pd.DataFrame) -> pd.DataFrame:
        """Merges name-based and address-based candidate pairs into unified candidate dataframe."""
        if nc.empty and ac.empty:
            return pd.DataFrame(columns=["s1_id", "matched_id", "name_score", "addr_score", "source"])

        nc["s1_id"] = nc["s1_id"].astype(str)
        nc["matched_id"] = nc["matched_id"].astype(str)
        ac["s1_id"] = ac["s1_id"].astype(str)
        ac["matched_id"] = ac["matched_id"].astype(str)

        merged = pd.merge(nc, ac, on=["s1_id", "matched_id"], how="outer")
        merged["name_score"] = merged["name_score"].fillna(0.0).astype(np.float32)
        merged["addr_score"] = merged["addr_score"].fillna(0.0).astype(np.float32)
        merged["source"] = np.where(merged["matched_id"].str.startswith("S2"), "S2", "S3")

        merged["s1_id"] = merged["s1_id"].astype("category")
        merged["matched_id"] = merged["matched_id"].astype("category")
        return merged
