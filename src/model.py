"""
LightGBM Entity Matching Ranker & Classifier Engine
==================================================
Provides GroupKFold-partitioned GBDT training, hard negative mining,
cross-validation scoring, out-of-fold probability estimation, and model persistence.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

from src.config import Config, default_config

logger = logging.getLogger(__name__)


class EntityMatchingModel:
    """
    Ensemble of LightGBM binary classifiers trained with GroupKFold by entity query ID.
    Prevents query leakage and produces calibrated out-of-fold probabilities.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or default_config
        self.models: List[lgb.Booster] = []
        self.feature_names: List[str] = []
        self.oof_predictions: Optional[np.ndarray] = None

    def train_cv(
        self,
        features_df: pd.DataFrame,
        labels: np.ndarray,
        groups: np.ndarray,
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Trains an ensemble across GroupKFold splits.
        Returns (oof_probabilities, validation_losses).
        """
        self.feature_names = [c for c in features_df.columns if c not in ("s1_id", "matched_id", "label", "prob")]
        n_splits = self.config.n_folds
        gkf = GroupKFold(n_splits=n_splits)

        oof_probs = np.zeros(len(features_df), dtype=np.float32)
        val_losses: List[float] = []
        self.models = []

        params = dict(self.config.lgbm_params)

        for fold, (train_idx, val_idx) in enumerate(gkf.split(features_df, groups=groups)):
            logger.info("Training fold %d / %d ...", fold + 1, n_splits)
            X_train = features_df.iloc[train_idx][self.feature_names]
            y_train = labels[train_idx]
            X_val = features_df.iloc[val_idx][self.feature_names]
            y_val = labels[val_idx]

            dtrain = lgb.Dataset(X_train, label=y_train)
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

            booster = lgb.train(
                params,
                dtrain,
                valid_sets=[dtrain, dval],
                callbacks=[
                    lgb.early_stopping(self.config.early_stopping_rounds, verbose=False),
                    lgb.log_evaluation(0),
                ],
            )
            self.models.append(booster)

            preds = booster.predict(X_val, num_iteration=booster.best_iteration).astype(np.float32)
            oof_probs[val_idx] = preds

            # Logloss computation
            eps = 1e-15
            clipped_preds = np.clip(preds, eps, 1.0 - eps)
            loss = float(-np.mean(y_val * np.log(clipped_preds) + (1.0 - y_val) * np.log(1.0 - clipped_preds)))
            val_losses.append(loss)
            logger.info("  Fold %d Best Iter: %d, Val Logloss: %.5f", fold + 1, booster.best_iteration, loss)

        self.oof_predictions = oof_probs
        return oof_probs, val_losses

    def predict_proba(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Computes ensemble average predicted probability of match across all fold models.
        """
        if not self.models:
            raise RuntimeError("Cannot predict with untrained model ensemble.")

        X = features_df[self.feature_names]
        ensemble_preds = np.zeros(len(X), dtype=np.float32)

        for booster in self.models:
            ensemble_preds += booster.predict(
                X, num_iteration=booster.best_iteration
            ).astype(np.float32)

        ensemble_preds /= float(len(self.models))
        return ensemble_preds

    def get_feature_importances(self, importance_type: str = "gain") -> Dict[str, float]:
        """
        Aggregates average feature importance across all trained fold boosters.
        """
        if not self.models:
            return {}

        scores = np.zeros(len(self.feature_names), dtype=np.float64)
        for booster in self.models:
            scores += booster.feature_importance(importance_type=importance_type)
        scores /= len(self.models)

        return dict(sorted(zip(self.feature_names, scores), key=lambda x: -x[1]))

    def save(self, model_dir: str):
        """Saves model ensemble and feature configuration to directory."""
        os.makedirs(model_dir, exist_ok=True)
        meta = {
            "feature_names": self.feature_names,
            "n_models": len(self.models),
        }
        with open(os.path.join(model_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        for i, booster in enumerate(self.models):
            booster.save_model(os.path.join(model_dir, f"model_fold_{i}.txt"))

    def load(self, model_dir: str) -> "EntityMatchingModel":
        """Loads saved ensemble from directory."""
        with open(os.path.join(model_dir, "metadata.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.feature_names = meta["feature_names"]
        n_models = meta["n_models"]

        self.models = []
        for i in range(n_models):
            path = os.path.join(model_dir, f"model_fold_{i}.txt")
            booster = lgb.Booster(model_file=path)
            self.models.append(booster)

        return self
