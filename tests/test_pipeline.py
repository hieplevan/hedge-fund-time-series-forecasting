import unittest
import numpy as np
import pandas as pd
from src.data.proxy_encoder import ProxyTargetEncoder
from src.features.engineering import engineer_features
from src.models.metrics import weighted_rmse_score, audit_forecast_distribution
from src.models.trainer import HorizonModelTrainer


class TestHedgeFundPipeline(unittest.TestCase):
    def test_weighted_rmse_perfect(self):
        """Kiem tra diem toi da 1.0 khi du bao khop hoan toan."""
        y = np.array([1.0, 2.0, 3.0, 4.0])
        w = np.array([1.0, 1.0, 1.0, 1.0])
        score = weighted_rmse_score(y, y, w)
        self.assertTrue(np.isclose(score, 1.0, atol=1e-5))

    def test_weighted_rmse_zeros(self):
        """Kiem tra xu ly an toan khi sai so lon hoac mau so khong duong."""
        y = np.array([0.0, 0.0])
        w = np.array([1.0, 1.0])
        self.assertEqual(weighted_rmse_score(y, y, w), 0.0)

        y_true = np.array([1.0, 1.0])
        y_pred = np.array([100.0, 100.0])
        self.assertEqual(weighted_rmse_score(y_true, y_pred, w), 0.0)

    def test_proxy_target_encoder(self):
        """Kiem tra encoder bo qua cac mau sau nguong validation."""
        df = pd.DataFrame({
            "ts_index": [100, 200, 300, 5000],
            "sub_category": ["A", "A", "B", "B"],
            "sub_code": ["X", "X", "Y", "Y"],
            "feature_al": [10.0, 20.0, 30.0, 999.0],
        })
        encoder = ProxyTargetEncoder(val_threshold=3500, proxy_col="feature_al", ts_col="ts_index")
        encoder.fit(df)

        self.assertTrue(np.isclose(encoder.stats["sub_category"]["A"], 15.0))
        self.assertTrue(np.isclose(encoder.stats["sub_category"]["B"], 30.0))

        transformed = encoder.transform(df)
        self.assertIn("sub_cat_proxy_enc", transformed.columns)
        self.assertIn("sub_code_proxy_enc", transformed.columns)
        self.assertEqual(transformed["sub_cat_proxy_enc"].iloc[0], 15.0)

    def test_embargo_split_gap(self):
        """Kiem tra vung dem cach ly delta giua train va val."""
        trainer = HorizonModelTrainer(lgb_params={}, val_threshold=3500)
        df = pd.DataFrame({"ts_index": np.arange(3400, 3600)})

        fit_mask, val_mask = trainer.get_embargo_split(df, hz=1)
        max_fit_ts = df.loc[fit_mask, "ts_index"].max()
        min_val_ts = df.loc[val_mask, "ts_index"].min()

        self.assertLessEqual(max_fit_ts, 3474)
        self.assertGreaterEqual(min_val_ts, 3501)
        self.assertGreaterEqual(min_val_ts - max_fit_ts, 27)

    def test_feature_engineering_pipeline(self):
        """Kiem tra tao dac trung tren du lieu chuoi thoi gian."""
        n_rows = 50
        synthetic_data = pd.DataFrame({
            "code": ["ASSET_1"] * n_rows,
            "sub_code": ["S1"] * n_rows,
            "sub_category": ["C1"] * n_rows,
            "ts_index": np.arange(1, n_rows + 1),
            "feature_al": np.linspace(10, 20, n_rows),
            "feature_am": np.linspace(5, 15, n_rows),
            "feature_cg": np.sin(np.linspace(0, 10, n_rows)),
            "feature_by": np.cos(np.linspace(0, 10, n_rows)),
            "feature_s": np.random.randn(n_rows),
            "feature_u": np.random.randn(n_rows),
            "feature_ac": np.random.randn(n_rows),
            "feature_ab": np.random.randn(n_rows),
            "feature_m": np.random.randn(n_rows),
            "feature_l": np.random.randn(n_rows),
        })

        enc_stats = {
            "sub_category": {"C1": 15.0},
            "sub_code": {"S1": 15.0},
            "global_mean": 15.0,
        }

        feats_h1 = engineer_features(synthetic_data, enc_stats, hz=1)
        self.assertIn("d_al_am", feats_h1.columns)
        self.assertIn("feature_al_cs", feats_h1.columns)
        self.assertIn("feature_al_wavelet_short", feats_h1.columns)
        self.assertIn("feature_al_wavelet_medium", feats_h1.columns)
        self.assertIn("feature_al_roc", feats_h1.columns)
        self.assertIn("feature_al_accel", feats_h1.columns)
        self.assertIn("al_proxy_spread", feats_h1.columns)

    def test_forecast_distribution_audit(self):
        """Kiem tra canh bao sup do phuong sai."""
        preds_normal = np.random.normal(loc=0.0, scale=1.0, size=1000)
        stats_norm = audit_forecast_distribution(preds_normal, zero_threshold=0.60)
        self.assertLess(stats_norm["zero_rate"], 0.10)

        preds_collapsed = np.zeros(1000)
        stats_collapsed = audit_forecast_distribution(preds_collapsed, zero_threshold=0.60)
        self.assertEqual(stats_collapsed["zero_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
