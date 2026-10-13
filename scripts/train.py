import argparse
import gc
import os
import sys
import time
from typing import Dict, List
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.loader import find_dataset_path, load_parquet_data
from src.data.proxy_encoder import ProxyTargetEncoder
from src.features.engineering import engineer_features
from src.models.metrics import weighted_rmse_score, audit_forecast_distribution
from src.models.trainer import HorizonModelTrainer
from src.models.inference import SequentialInferencePipeline
from src.utils.helpers import get_logger, load_config, seed_everything

logger = get_logger("TrainPipeline")


def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline du bao chuoi thoi gian Hedge Fund")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Duong dan file cau hinh YAML")
    parser.add_argument("--data-dir", type=str, default=None, help="Duong dan thu muc du lieu")
    parser.add_argument("--skip-inference", action="store_true", help="Bo qua buoc suy luan test")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["model"]["seeds"][0])

    data_dir = args.data_dir or find_dataset_path(cfg["data"]["candidate_dirs"])
    train_path = os.path.join(data_dir, "train.parquet")
    test_path = os.path.join(data_dir, "test.parquet")

    logger.info("=" * 65)
    logger.info("PIPELINE DU BAO CHUOI THOI GIAN HEDGE FUND")
    logger.info("=" * 65)
    logger.info(f"Train: {train_path}")
    logger.info(f"Test : {test_path}")

    target_col = cfg["columns"]["target"]
    weight_col = cfg["columns"]["weight"]
    id_col = cfg["columns"]["id"]
    ts_col = cfg["columns"]["ts_index"]
    hz_col = cfg["columns"]["horizon"]
    val_threshold = cfg["split"]["val_threshold"]
    history_tail = cfg["split"]["history_tail"]
    horizons = cfg["model"]["horizons"]

    # 1. Tinh Proxy Target Encoding tren nguong train
    logger.info("Tinh toan Proxy Target Encoding...")
    temp_enc = load_parquet_data(
        train_path,
        columns=["sub_category", "sub_code", "feature_al", ts_col],
    )
    encoder = ProxyTargetEncoder(val_threshold=val_threshold, proxy_col="feature_al", ts_col=ts_col)
    encoder.fit(temp_enc)
    proxy_stats = encoder.stats
    del temp_enc
    gc.collect()

    trainer = HorizonModelTrainer(
        lgb_params=cfg["model"]["lgb_params"],
        seeds=cfg["model"]["seeds"],
        val_threshold=val_threshold,
        early_stopping_rounds=cfg["model"]["early_stopping_rounds"],
    )

    horizon_models = {}
    horizon_features = {}
    horizon_tail = {}
    cv_cache: Dict[str, List[float]] = {"y": [], "pred": [], "wt": []}

    exclude_cols = {
        id_col, "code", "sub_code", "sub_category", hz_col, ts_col,
        weight_col, target_col, "weight_adjusted",
    }

    start_time = time.time()

    # 2. Huan luyen mo hinh theo tung Horizon voi Dynamic Embargo
    for hz in horizons:
        t_hz = time.time()
        logger.info(f"\n--- HORIZON = {hz} ---")
        raw_train = load_parquet_data(train_path, filters=[(hz_col, "==", hz)])

        max_ts = int(raw_train[ts_col].max())
        horizon_tail[hz] = raw_train[raw_train[ts_col] > max_ts - history_tail].copy()

        logger.info(f"Trich xuat dac trung cho Horizon {hz}...")
        df_feats = engineer_features(raw_train, proxy_stats, hz, ts_col=ts_col)
        del raw_train
        gc.collect()

        feats = [c for c in df_feats.columns if c not in exclude_cols]
        horizon_features[hz] = feats

        fit_mask, val_mask = trainer.get_embargo_split(df_feats, hz, ts_col=ts_col)
        X_fit = df_feats.loc[fit_mask, feats]
        y_fit = df_feats.loc[fit_mask, target_col]
        w_fit = df_feats.loc[fit_mask, weight_col].astype(float)

        X_val = df_feats.loc[val_mask, feats]
        y_val = df_feats.loc[val_mask, target_col]
        w_val = df_feats.loc[val_mask, weight_col].astype(float)

        logger.info(f"Phan chia: Fit={len(X_fit):,} | Val={len(X_val):,} | Features={len(feats)}")

        models = trainer.train_seed_ensemble(X_fit, y_fit, w_fit, X_val, y_val, w_val)
        horizon_models[hz] = models

        val_preds = trainer.ensemble_predict(models, X_val)
        val_score = weighted_rmse_score(y_val, val_preds, w_val)
        logger.info(f"Validation Score [H={hz}]: {val_score:.5f} (Thoi gian: {(time.time() - t_hz)/60:.1f}m)")

        cv_cache["y"].extend(y_val.tolist())
        cv_cache["pred"].extend(val_preds.tolist())
        cv_cache["wt"].extend(w_val.tolist())

        del df_feats, X_fit, y_fit, w_fit, X_val, y_val, w_val
        gc.collect()

    overall_score = weighted_rmse_score(cv_cache["y"], cv_cache["pred"], cv_cache["wt"])
    logger.info("=" * 65)
    logger.info(f"DIEM OUT-OF-SAMPLE VALIDATION TONG: {overall_score:.5f}")
    logger.info(f"Tong thoi gian huan luyen: {(time.time() - start_time)/60:.1f} phut")
    logger.info("=" * 65)

    # 3. Suy luan tuan tu tren tap test
    if not args.skip_inference:
        logger.info("\nBat dau suy luan tren tap test...")
        t_inf = time.time()
        infer_pipeline = SequentialInferencePipeline(id_col=id_col, ts_col=ts_col)
        test_outputs = []

        for hz in horizons:
            t_hz_inf = time.time()
            raw_test = load_parquet_data(test_path, filters=[(hz_col, "==", hz)])
            preds = infer_pipeline.predict_horizon(
                test_df_hz=raw_test,
                train_tail=horizon_tail[hz],
                models=horizon_models[hz],
                features=horizon_features[hz],
                enc_stats=proxy_stats,
                hz=hz,
            )
            test_outputs.append(preds)
            logger.info(f"Hoan tat H={hz}: {len(preds):,} dong ({(time.time()-t_hz_inf)/60:.1f}m)")
            del raw_test
            gc.collect()

        final_sub = pd.concat(test_outputs, ignore_index=True)
        logger.info(f"Tong thoi gian suy luan: {(time.time() - t_inf)/60:.1f}m")

        # Kiem tra phan phoi du bao
        for hz in horizons:
            sub_hz = final_sub[final_sub["horizon"] == hz]["prediction"].values
            audit_forecast_distribution(
                sub_hz,
                atol=cfg["diagnostics"]["zero_tolerance"],
                zero_threshold=cfg["diagnostics"]["zero_rate_threshold"],
                horizon_name=f"Horizon {hz}",
            )

        submission = final_sub[[id_col, "prediction"]].copy()
        infer_pipeline.validate_submission(submission)

        sub_out = cfg["data"]["submission_path"]
        submission.to_csv(sub_out, index=False)
        logger.info(f"Luu submission thanh cong vao {sub_out} (shape={submission.shape})")
        logger.info(f"\n{submission.head(5).to_string(index=False)}")


if __name__ == "__main__":
    main()
