from typing import Any, Dict, List
import numpy as np
import pandas as pd
from src.features.engineering import engineer_features
from src.models.trainer import HorizonModelTrainer
from src.utils.helpers import get_logger

logger = get_logger("Inference")


class SequentialInferencePipeline:
    """Suy luan tuan tu bang cach ghep train_tail vao test set de giu lien tuc cac bien lag/rolling."""

    def __init__(
        self,
        id_col: str = "id",
        ts_col: str = "ts_index",
    ):
        self.id_col = id_col
        self.ts_col = ts_col

    def predict_horizon(
        self,
        test_df_hz: pd.DataFrame,
        train_tail: pd.DataFrame,
        models: List[Any],
        features: List[str],
        enc_stats: Dict[str, Any],
        hz: int,
    ) -> pd.DataFrame:
        """Ghep train_tail voi tap test de tinh dac trung va du bao."""
        max_train_ts = int(train_tail[self.ts_col].max())
        tail_cols = [c for c in train_tail.columns if c in test_df_hz.columns]

        combined = pd.concat([train_tail[tail_cols], test_df_hz], ignore_index=True)
        engineered = engineer_features(combined, enc_stats, hz, ts_col=self.ts_col)

        te_feats = engineered[engineered[self.ts_col] > max_train_ts].reset_index(drop=True)

        for f in features:
            if f not in te_feats.columns:
                te_feats[f] = np.nan

        preds = HorizonModelTrainer.ensemble_predict(models, te_feats[features])

        result = te_feats[[self.id_col]].copy()
        result["prediction"] = preds
        result["horizon"] = hz
        return result

    def validate_submission(self, submission_df: pd.DataFrame) -> None:
        """Kiem tra tinh hop le cua file submission."""
        if self.id_col not in submission_df.columns:
            raise ValueError(f"Thieu cot ID: {self.id_col}")
        if "prediction" not in submission_df.columns:
            raise ValueError("Thieu cot prediction trong submission")
        if submission_df[self.id_col].nunique() != len(submission_df):
            raise ValueError("Phat hien ID bi trung lap trong submission")
        if submission_df["prediction"].isna().any():
            raise ValueError("Phat hien gia tri NaN trong cot prediction")

        logger.info("Kiem tra file submission thanh cong.")
