from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from src.utils.helpers import get_logger

logger = get_logger("ProxyTargetEncoder")


class ProxyTargetEncoder:
    """Ma hoa bien dinh danh dua tren tin hieu dan dat feature_al, tranh look-ahead bias."""

    def __init__(
        self,
        val_threshold: int = 3500,
        proxy_col: str = "feature_al",
        ts_col: str = "ts_index",
    ):
        self.val_threshold = val_threshold
        self.proxy_col = proxy_col
        self.ts_col = ts_col
        self.stats: Dict[str, Any] = {}

    def fit(self, df: pd.DataFrame) -> "ProxyTargetEncoder":
        """Tinh trung binh feature_al truoc nguong validation de tranh ro ri du lieu."""
        cols_needed = ["sub_category", "sub_code", self.proxy_col, self.ts_col]
        missing = [c for c in cols_needed if c not in df.columns]
        if missing:
            raise KeyError(f"Thieu cac cot can thiet: {missing}")

        train_enc_rows = df[df[self.ts_col] <= self.val_threshold]
        if train_enc_rows.empty:
            raise ValueError(f"Khong tim thay ban ghi voi {self.ts_col} <= {self.val_threshold}")

        global_mean = float(train_enc_rows[self.proxy_col].mean())
        sub_cat_stats = train_enc_rows.groupby("sub_category")[self.proxy_col].mean().to_dict()
        sub_code_stats = train_enc_rows.groupby("sub_code")[self.proxy_col].mean().to_dict()

        self.stats = {
            "sub_category": sub_cat_stats,
            "sub_code": sub_code_stats,
            "global_mean": global_mean,
        }

        logger.info(
            f"Proxy encoding fit (<= {self.val_threshold}): "
            f"{len(sub_cat_stats)} sub_categories, {len(sub_code_stats)} sub_codes"
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Anh xa gia tri thong ke da tinh vao dataframe."""
        if not self.stats:
            raise RuntimeError("Can goi fit truoc khi transform.")

        data = df.copy()
        global_mean = self.stats["global_mean"]

        if "sub_category" in data.columns:
            data["sub_cat_proxy_enc"] = (
                data["sub_category"]
                .astype(str)
                .map(self.stats["sub_category"])
                .fillna(global_mean)
                .astype(np.float32)
            )

        if "sub_code" in data.columns:
            data["sub_code_proxy_enc"] = (
                data["sub_code"]
                .astype(str)
                .map(self.stats["sub_code"])
                .fillna(global_mean)
                .astype(np.float32)
            )

        return data

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
