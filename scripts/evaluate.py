import argparse
import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.metrics import audit_forecast_distribution
from src.utils.helpers import get_logger, load_config

logger = get_logger("Evaluator")


def parse_args():
    parser = argparse.ArgumentParser(description="Danh gia phan phoi du bao")
    parser.add_argument("--submission", type=str, default="submission.csv", help="Duong dan file submission.csv")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Duong dan file cau hinh YAML")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if not os.path.exists(args.submission):
        logger.error(f"Khong tim thay file submission: {args.submission}")
        return

    df_sub = pd.read_csv(args.submission)
    logger.info(f"Nap submission tu {args.submission} (shape={df_sub.shape})")

    if "prediction" not in df_sub.columns:
        logger.error("Thieu cot prediction trong submission CSV")
        return

    preds = df_sub["prediction"].values
    audit_forecast_distribution(
        preds,
        atol=cfg["diagnostics"]["zero_tolerance"],
        zero_threshold=cfg["diagnostics"]["zero_rate_threshold"],
        horizon_name="Tong the Submission",
    )


if __name__ == "__main__":
    main()
