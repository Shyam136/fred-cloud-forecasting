
import unittest
import pandas as pd
from scripts.features.feature_builder import build_target_features

class TestFeatureBuilder(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range('2023-01-01', periods=24, freq='MS')
        values = pd.Series(range(100, 124))  # simple increasing
        self.df = pd.DataFrame({'date': dates, 'CPIAUCSL': values})

    def test_basic_columns(self):
        feats = build_target_features(self.df, 'CPIAUCSL', lags=[1,3], ma_windows=[3], std_windows=[3])
        expected = {'date', 'CPIAUCSL', 'CPIAUCSL_lag1', 'CPIAUCSL_lag3',
                    'CPIAUCSL_ma3', 'CPIAUCSL_std3', 'CPIAUCSL_pct_change'}
        self.assertTrue(expected.issubset(set(feats.columns)))
        self.assertEqual(len(feats), len(self.df))

    def test_lag_values(self):
        feats = build_target_features(self.df, 'CPIAUCSL', lags=[1], ma_windows=[], std_windows=[])
        # at index 5, lag1 equals original index 4
        self.assertEqual(feats.loc[5, 'CPIAUCSL_lag1'], feats.loc[4, 'CPIAUCSL'])

    def test_pct_change(self):
        feats = build_target_features(self.df, 'CPIAUCSL', lags=[], ma_windows=[], std_windows=[], add_pct_change=True)
        self.assertTrue(pd.notna(feats.loc[1, 'CPIAUCSL_pct_change']))

if __name__ == '__main__':
    unittest.main()