import os
import time
from typing import Any, List, Optional
import numpy as np
import pandas as pd
from src.utils.helpers import get_logger

logger = get_logger("DataLoader")


def find_dataset_path(candidate_paths: Optional[List[str]] = None) -> str:
    """Tu dong tim duong dan chua tap train.parquet."""
    if candidate_paths is None:
        candidate_paths = [
            "/kaggle/input/competitions/ts-forecasting",
            "/kaggle/input/ts-forecasting",
            "/kaggle/input/hedge-fund-time-series",
            "data/raw",
            ".",
        ]

    for path in candidate_paths:
        if os.path.exists(os.path.join(path, "train.parquet")):
            logger.info(f"Tim thay du lieu tai: {path}")
            return path

    if os.path.exists("/kaggle/input"):
        for root, _, files in os.walk("/kaggle/input"):
            if "train.parquet" in files:
                logger.info(f"Tim thay qua tim kiem: {root}")
                return root

    raise FileNotFoundError("Khong tim thay train.parquet trong cac duong dan.")


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Giam dung luong RAM bang cach ep kieu float32 va int32."""
    start_mem = df.memory_usage().sum() / 1024**2 if verbose else 0
    for col in df.columns:
        dtype = df[col].dtype
        if dtype != object and dtype.name != "category":
            if dtype == np.float64:
                df[col] = df[col].astype(np.float32)
            elif dtype == np.int64:
                df[col] = df[col].astype(np.int32)
    if verbose:
        end_mem = df.memory_usage().sum() / 1024**2
        logger.info(f"Giam RAM: {start_mem:.2f} MB -> {end_mem:.2f} MB")
    return df


def load_parquet_data(
    file_path: str,
    columns: Optional[List[str]] = None,
    filters: Optional[List[Any]] = None,
) -> pd.DataFrame:
    """Doc file parquet toi uu bo nho."""
    t0 = time.time()
    df = pd.read_parquet(file_path, columns=columns, filters=filters)
    logger.info(f"Nap {file_path} thanh cong (shape={df.shape}, thoi gian={time.time()-t0:.2f}s)")
    return df
