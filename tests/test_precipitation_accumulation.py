import unittest

import numpy as np
import pandas as pd

from src.data.preprocess_era5land import deaccumulate_precipitation


class AccumulationTests(unittest.TestCase):
    def test_daily_total_and_reset(self):
        times = pd.date_range("2000-01-01 01:00", periods=25, freq="h")
        cumulative = np.array(list(range(1, 25)) + [1])[:, None] * 0.001
        hourly = deaccumulate_precipitation(times, cumulative)
        np.testing.assert_allclose(hourly, 0.001)
        self.assertAlmostEqual(hourly[:24].sum(), cumulative[23, 0])

    def test_month_boundary_and_missing_predecessor(self):
        times = pd.date_range("2000-02-01", periods=2, freq="h")
        values = np.array([[0.024], [0.001]])
        self.assertTrue(np.isnan(deaccumulate_precipitation(times, values)[0, 0]))
        result = deaccumulate_precipitation(times, values, times[0] - pd.Timedelta(hours=1), np.array([0.023]))
        np.testing.assert_allclose(result, 0.001)

    def test_gap_and_negatives(self):
        times = pd.to_datetime(["2000-01-01 01:00", "2000-01-01 03:00"])
        self.assertTrue(np.isnan(deaccumulate_precipitation(times, [[0.001], [0.003]])[1, 0]))
        times = pd.date_range("2000-01-01 01:00", periods=2, freq="h")
        np.testing.assert_allclose(deaccumulate_precipitation(times, [[0.001], [0.001 - 1e-9]])[1], 0)
        with self.assertRaises(ValueError):
            deaccumulate_precipitation(times, [[0.002], [0.001]])
        with self.assertRaises(ValueError):
            deaccumulate_precipitation([times[0], times[0]], [[0], [0]])
