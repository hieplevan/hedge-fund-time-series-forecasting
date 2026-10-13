import os
import time
from typing import Any, Dict, List, Optional, Tuple
import lightgbm as lgb
import numpy as np
import pandas as pd
from src.models.metrics import weighted_rmse_score
from src.utils.helpers import get_logger

logger = get_logger("ModelTrainer")


class HorizonModelTrainer:
    """Huan luyen mo hinh LightGBM voi Dynamic Embargo va Seed Blending."""

    def __init__(
        self,
        lgb_params: Dict[str, Any],
        seeds: Optional[List[int]] = None,
        val_threshold: int = 3500,
        early_stopping_rounds: int = 100,
    ):
        self.lgb_params = lgb_params
        self.seeds = seeds if seeds is not None else [42, 2024, 12345, 99, 420]
        self.val_threshold = val_threshold
        self.early_stopping_rounds = early_stopping_rounds

    def get_embargo_split(
        self, df: pd.DataFrame, hz: int, ts_col: str = "ts_index"
    ) -> Tuple[pd.Series, pd.Series]:
        """Tao vung dem cach ly delta = H + 25 de triet tieu Target Overlap Leakage."""
        if hz in [1, 10]:
            embargo_limit = self.val_threshold - hz - 25
        elif hz == 3:
            embargo_limit = self.val_threshold - 25
        else:
            embargo_limit = self.val_threshold

        fit_mask = df[ts_col] <= embargo_limit
        val_mask = df[ts_col] > self.val_threshold

        logger.info(
            f"[Horizon {hz}] Nguong Embargo: {embargo_limit} "
            f"(Val: {self.val_threshold}, Vung dem: {self.val_threshold - embargo_limit} buoc)"
        )
        return fit_mask, val_mask

    def train_seed_ensemble(
        self,
        X_fit: pd.DataFrame,
        y_fit: pd.Series,
        w_fit: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        w_val: pd.Series,
    ) -> List[lgb.LGBMRegressor]:
        """Huan luyen ensemble nhieu seeds de giam variance."""
        models: List[lgb.LGBMRegressor] = []
        for seed in self.seeds:
            cfg = {**self.lgb_params, "random_state": seed}
            model = lgb.LGBMRegressor(**cfg)
            callbacks = [lgb.early_stopping(self.early_stopping_rounds, verbose=False)]

            model.fit(
                X_fit,
                y_fit,
                sample_weight=w_fit,
                eval_set=[(X_val, y_val)],
                eval_sample_weight=[w_val],
                callbacks=callbacks,
            )
            models.append(model)
        return models

    @staticmethod
    def ensemble_predict(
        models: List[lgb.LGBMRegressor], X: pd.DataFrame
    ) -> np.ndarray:
        """Du bao trung binh cong tu cac mo hinh ensemble."""
        preds = np.zeros(len(X), dtype=np.float64)
        for m in models:
            preds += m.predict(X) / len(models)
        return preds
