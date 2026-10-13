from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from src.data.loader import reduce_mem_usage
from src.utils.helpers import get_logger

logger = get_logger("FeatureEngineering")


def calc_slope(y: np.ndarray) -> float:
    """Tinh do doc tuyen tinh (slope) tren cua so truot."""
    if len(y) >= 2:
        return float(np.polyfit(np.arange(len(y)), y, 1)[0])
    return np.nan


def engineer_features(
    data: pd.DataFrame,
    enc_stats: Dict[str, Any],
    hz: int,
    ts_col: str = "ts_index",
) -> pd.DataFrame:
    """
    Trich xuat dac trung chuoi thoi gian da thang do:
      1. Anh xa Proxy Encoding (prior knowledge)
      2. Chenh lech va ty le giua cac cap bien
      3. Chuan hoa Cross-sectional Z-score theo ts_index (khu Market Beta)
      4. Bo loc Pseudo-Wavelet (EMA 3, 10, 25, 50)
      5. Dong hoc bac hai (Gia toc, ROC, Proxy Spread)
      6. Cac dac trung chuyen biet theo tung Horizon
    """
    x = data.copy()
    if "code" in x.columns and ts_col in x.columns:
        x = x.sort_values(by=["code", ts_col]).reset_index(drop=True)

    # 1. Ep kieu category va anh xa Proxy Target Encoding
    for c in ["code", "sub_code", "sub_category"]:
        if c in x.columns:
            x[c] = x[c].astype("category")

    global_mean = enc_stats.get("global_mean", 0.0)
    if "sub_category" in x.columns:
        x["sub_cat_proxy_enc"] = (
            x["sub_category"]
            .astype(str)
            .map(enc_stats.get("sub_category", {}))
            .fillna(global_mean)
            .astype(np.float32)
        )

    if "sub_code" in x.columns:
        x["sub_code_proxy_enc"] = (
            x["sub_code"]
            .astype(str)
            .map(enc_stats.get("sub_code", {}))
            .fillna(global_mean)
            .astype(np.float32)
        )

    # 2. Chenh lech va ty le giua cac dac trung
    if "feature_u" in x.columns and "feature_s" in x.columns:
        x["d_u_s"] = (x["feature_u"] - x["feature_s"]).astype(np.float32)
    if "feature_ac" in x.columns and "feature_ab" in x.columns:
        x["d_ac_ab"] = (x["feature_ac"] - x["feature_ab"]).astype(np.float32)
    if "feature_m" in x.columns and "feature_l" in x.columns:
        x["d_m_l"] = (x["feature_m"] - x["feature_l"]).astype(np.float32)
    if "feature_al" in x.columns and "feature_am" in x.columns:
        x["d_al_am"] = (x["feature_al"] - x["feature_am"]).astype(np.float32)
        x["r_al_am"] = (x["feature_al"] / (x["feature_am"] + 1e-7)).astype(np.float32)
    if "feature_cg" in x.columns and "feature_by" in x.columns:
        x["d_cg_by"] = (x["feature_cg"] - x["feature_by"]).astype(np.float32)

    # 3. Chuan hoa Cross-sectional Z-score (khu bien dong vi mo chung toan thi truong)
    norm_cols = [c for c in ["feature_al", "feature_am", "feature_cg", "feature_by", "d_al_am"] if c in x.columns]
    for col in norm_cols:
        g = x.groupby(ts_col)[col]
        x[f"{col}_cs"] = ((x[col] - g.transform("mean")) / (g.transform("std") + 1e-7)).astype(np.float32)

    cs_cols = [f"{c}_cs" for c in norm_cols if f"{c}_cs" in x.columns]
    if cs_cols:
        x["outlier_count"] = (x[cs_cols].abs() > 3.0).sum(axis=1).astype(np.int8)

    # 4. Dong hoc va Pseudo-Wavelet tren cac tai san dan dat
    top_features = ["feature_al", "feature_am", "feature_cg", "feature_by", "feature_s"]
    for col in top_features:
        if col not in x.columns:
            continue

        grp = x.groupby("code", observed=True)[col]

        # Lags da chu ky
        for lag in [1, 3, 5, 10, 25]:
            x[f"{col}_lag{lag}"] = grp.shift(lag).astype(np.float32)

        # Sai phan diff1 va cac duong EMA
        x[f"{col}_diff1"] = (x[col] - x[f"{col}_lag1"]).astype(np.float32)
        x[f"{col}_ema3"] = grp.transform(lambda s: s.ewm(span=3, adjust=False).mean()).astype(np.float32)
        x[f"{col}_ema10"] = grp.transform(lambda s: s.ewm(span=10, adjust=False).mean()).astype(np.float32)
        x[f"{col}_ema25"] = grp.transform(lambda s: s.ewm(span=25, adjust=False).mean()).astype(np.float32)

        # Pseudo-Wavelets: Bo loc dai thong bóc tach vi pha va xu huong
        x[f"{col}_wavelet_short"] = (x[f"{col}_ema3"] - x[f"{col}_ema10"]).astype(np.float32)
        x[f"{col}_wavelet_medium"] = (x[f"{col}_ema10"] - x[f"{col}_ema25"]).astype(np.float32)

        # Chenh lech so voi trung binh phan nganh sub_code
        if "sub_code" in x.columns:
            mean_sub = x.groupby([ts_col, "sub_code"], observed=True)[col].transform("mean")
            x[f"{col}_vs_subcode"] = (x[col] - mean_sub).astype(np.float32)

        # Do bien dong rolling std 10
        x[f"{col}_roll_std_10"] = grp.transform(lambda s: s.rolling(10, min_periods=1).std()).astype(np.float32)

        # Do doc rolling slope
        if col in ["feature_al", "feature_am"]:
            x[f"{col}_trend_10"] = grp.transform(
                lambda s: s.rolling(10, min_periods=2).apply(calc_slope, raw=True)
            ).astype(np.float32)

        # Mo rong dai han cho Horizon 25
        if hz == 25:
            x[f"{col}_lag50"] = grp.shift(50).astype(np.float32)
            x[f"{col}_ema50"] = grp.transform(lambda s: s.ewm(span=50, adjust=False).mean()).astype(np.float32)
            x[f"{col}_wavelet_long"] = (x[f"{col}_ema25"] - x[f"{col}_ema50"]).astype(np.float32)
            if col in ["feature_al", "feature_am"]:
                x[f"{col}_trend_25"] = grp.transform(
                    lambda s: s.rolling(25, min_periods=2).apply(calc_slope, raw=True)
                ).astype(np.float32)

        # Dong hoc bac hai va Proxy Spread cho feature_al
        if col == "feature_al":
            # Toc do thay doi ROC (khu phu thuoc quy mo)
            x["feature_al_roc"] = (x["feature_al_diff1"] / (x["feature_al_lag1"].abs() + 1e-7)).astype(np.float32)

            # Gia toc (dao ham bac 2) don dau diem uon
            al_lag2 = grp.shift(2).astype(np.float32)
            x["feature_al_accel"] = (x["feature_al_diff1"] - (x["feature_al_lag1"] - al_lag2)).astype(np.float32)

            # Proxy Spread: Co lap alpha noi tai khoi ky vong nganh
            if "sub_code_proxy_enc" in x.columns:
                x["al_proxy_spread"] = (x["feature_al_roc"] - x["sub_code_proxy_enc"]).astype(np.float32)

            # Gia toc chuan hoa cross-sectional
            g_ts = x.groupby(ts_col)["feature_al_accel"]
            x["feature_al_accel_cs"] = (
                (x["feature_al_accel"] - g_ts.transform("mean")) / (g_ts.transform("std") + 1e-7)
            ).astype(np.float32)

    # 5. Dac trung chuyen biet theo tung Horizon
    if hz in [3, 10]:
        mom_cols = [
            c for c in [f"{c}_diff1" for c in top_features]
            + [f"{c}_wavelet_short" for c in top_features]
            + [f"{c}_wavelet_medium" for c in top_features]
            + ["feature_al_roc"]
            if c in x.columns
        ]
        for col in mom_cols:
            g = x.groupby(ts_col)[col]
            x[f"{col}_cs"] = ((x[col] - g.transform("mean")) / (g.transform("std") + 1e-7)).astype(np.float32)

        if hz == 10:
            all_cs = [c for c in x.columns if c.endswith("_cs")]
            if all_cs:
                x["outlier_count"] = (x[all_cs].abs() > 3.0).sum(axis=1).astype(np.int8)

    if hz == 1:
        x["missing_info_count"] = (
            x[x.select_dtypes(exclude=["category"]).columns].isna().sum(axis=1).astype(np.int8)
        )

    return reduce_mem_usage(x)


class FeaturePipeline:
    """Wrapper dong goi feature engineering."""

    def __init__(self, enc_stats: Dict[str, Any], horizon: int):
        self.enc_stats = enc_stats
        self.horizon = horizon

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return engineer_features(df, self.enc_stats, self.horizon)
