"""
Centralized Configuration Module for Multilingual Entity Resolution
===================================================================
Manages system parameters, dataset paths, blocking hyperparameters,
feature extraction settings, LightGBM model configuration, and output targets.
"""

import os
import sys
import json
import hashlib
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class Config:
    """Master configuration class."""

    # Execution Mode: 'smoke' (quick dry run), 'validation' (CV subset), 'full' (all data)
    mode: str = "full"

    # Paths
    dataset_root: str = ""
    train_s1_path: str = ""
    train_s2_path: str = ""
    train_s3_path: str = ""
    train_gt_path: str = ""
    test_s1_path: str = ""
    test_s2_path: str = ""
    test_s3_path: str = ""
    validator_path: str = ""
    output_dir: str = "output"
    cache_dir: str = "cache"

    # Blocking Hyperparameters
    k_name: int = 30
    k_addr: int = 10
    blocking_threshold: float = 0.20
    chunk_size: int = 500
    country_aware: bool = True
    max_features_char: int = 40000
    max_features_word: int = 40000

    # Model Hyperparameters (LightGBM)
    lgbm_params: Dict[str, Any] = field(default_factory=lambda: {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": 8,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "n_estimators": 800,
        "n_jobs": -1,
        "seed": 42,
        "verbose": -1,
    })
    n_folds: int = 3
    early_stopping_rounds: int = 30
    decision_threshold: float = 0.55

    # Post-Processing & Precision SOTA Settings
    min_name_similarity: float = 95.0
    min_addr_similarity: float = 70.0
    strict_singleton_threshold: float = 98.0
    max_matches_per_entity: int = 2

    def __post_init__(self):
        self.resolve_paths()

    def resolve_paths(self):
        """Auto-detects and resolves dataset and validator paths across environments."""
        candidates = [
            self.dataset_root,
            os.environ.get("DATASET_ROOT", ""),
            r"C:\Users\sivar\OneDrive\Desktop\student_resource",
            r"C:\Users\sivar\OneDrive\Desktop\student_resource\student_resource",
            "/kaggle/input/datasets/srkreddy6115/amazon-ml-challenge-2026/student_resource",
            "/kaggle/input/student_resource",
            "/home/ubuntu/student_resource",
        ]

        found_root = None
        for cand in candidates:
            if cand and os.path.isdir(cand):
                train_file = os.path.join(cand, "dataset", "train", "train_source1.tsv")
                if os.path.isfile(train_file):
                    found_root = cand
                    break

        if found_root:
            self.dataset_root = found_root
            td = os.path.join(found_root, "dataset", "train")
            tt = os.path.join(found_root, "dataset", "test")
            self.train_s1_path = os.path.join(td, "train_source1.tsv")
            self.train_s2_path = os.path.join(td, "train_source2.tsv")
            self.train_s3_path = os.path.join(td, "train_source3.tsv")
            self.train_gt_path = os.path.join(td, "train_ground_truth.tsv")
            self.test_s1_path = os.path.join(tt, "test_source1.tsv")
            self.test_s2_path = os.path.join(tt, "test_source2.tsv")
            self.test_s3_path = os.path.join(tt, "test_source3.tsv")
            self.validator_path = os.path.join(found_root, "utils", "validate_submission.py")

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

    @property
    def config_hash(self) -> str:
        """Computes deterministic hash for caching."""
        payload = {
            "k_name": self.k_name,
            "k_addr": self.k_addr,
            "threshold": self.blocking_threshold,
            "country_aware": self.country_aware,
        }
        return hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:8]


# Default global instance
default_config = Config()
