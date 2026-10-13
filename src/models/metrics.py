from typing import Dict, Union
import numpy as np
import pandas as pd
from src.utils.helpers import get_logger

logger = get_logger("Metrics")


def weighted_rmse_score(
    y_true: Union[np.ndarray, pd.Series, list],
    y_pred: Union[np.ndarray, pd.Series, list],
    weights: Union[np.ndarray, pd.Series, list],
) -> float:
    """Tinh Weighted RMSE score theo dung cong thuc cua cuoc thi."""
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)
    w_arr = np.asarray(weights, dtype=np.float64)

    denom = np.sum(w_arr * (y_true_arr**2))
    if denom <= 0.0:
        return 0.0

    ratio = np.sum(w_arr * ((y_true_arr - y_pred_arr) ** 2)) / denom
    clamped_ratio = float(np.clip(ratio, 0.0, 1.0))
    return float(np.sqrt(1.0 - clamped_ratio))


def audit_forecast_distribution(
    predictions: np.ndarray,
    atol: float = 1e-12,
    zero_threshold: float = 0.60,
    horizon_name: str = "Overall",
) -> Dict[str, float]:
    """Chan doan ty le du bao bang 0 de kiem soat sup do phuong sai (zero-inflation)."""
    total = len(predictions)
    if total == 0:
        return {"count": 0, "mean": 0.0, "std": 0.0, "zero_rate": 0.0}

    zero_mask = np.isclose(predictions, 0.0, atol=atol)
    zero_rate = float(np.mean(zero_mask))
    mean_val = float(np.mean(predictions))
    std_val = float(np.std(predictions))

    logger.info(
        f"[{horizon_name}] So mau: {total:,} | Mean: {mean_val:+.5f} | "
        f"Std: {std_val:.5f} | Ty le bang 0: {zero_rate * 100:.2f}%"
    )

    if zero_rate > zero_threshold:
        logger.warning(
            f"CANH BAO [{horizon_name}]: Ty le bang 0 ({zero_rate*100:.1f}%) vuot qua "
            f"nguong ({zero_threshold*100:.1f}%). Nguy co sup do phuong sai!"
        )

    return {
        "count": total,
        "mean": mean_val,
        "std": std_val,
        "zero_rate": zero_rate,
    }
