import unittest
import shutil
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from gofer.temporal_downsampler import aggregate


TEST_DATA_DIR = Path("tests/gofer/data/temporal")
TEMP_DIR = TEST_DATA_DIR / "hourly_output"
CSV_PATH = TEST_DATA_DIR / "test_files.csv"
GOES_SAVE_DIR = "tests/gofer/data"

# 5 hourly dates: hours 1-4 have or should have data, hour 3 is a gap
DATES = pd.date_range("2020-09-06 20:00", periods=5, freq="h")


class TestTemporalDownsamplerPerimeter(unittest.TestCase):
    """
    Test aggregate() in perimeter mode (is_perimeter=True).

    Scenario: 8 observations across 4 hours (2 per hour) with a 1-hour gap
    at hour 3. Fire grows from 2x2 -> 3x3 -> gap -> 4x4 -> 4x4 (unchanged).
    """

    @classmethod
    def setUpClass(cls):
        cls.ds = aggregate(
            goes_save_dir=GOES_SAVE_DIR,
            csv_path=str(CSV_PATH),
            temp_dir=str(TEMP_DIR),
            dates=DATES,
            data_var='MaskConfidence',
            fire_name='test_fire',
            is_perimeter=True,
            verbose=False,
        )

    @classmethod
    def tearDownClass(cls):
        """Clean up intermediate hourly nc files."""
        if TEMP_DIR.exists():
            shutil.rmtree(TEMP_DIR)

    def test_output_has_correct_time_length(self):
        """Should have 5 timesteps matching the input dates."""
        self.assertEqual(len(self.ds.time), 5)

    def test_output_times_match_expected_dates(self):
        """Output times should match the requested hourly dates."""
        expected = pd.DatetimeIndex(DATES.tz_localize(None))
        actual = pd.DatetimeIndex(self.ds.time.values)
        self.assertTrue(expected.equals(actual))

    def test_output_has_mask_confidence(self):
        """Output should contain MaskConfidence variable."""
        self.assertIn('MaskConfidence', self.ds.data_vars)

    def test_spatial_dims_are_10x10(self):
        """Spatial dimensions should be preserved at 10x10."""
        self.assertEqual(self.ds.sizes['y'], 10)
        self.assertEqual(self.ds.sizes['x'], 10)

    def test_no_nans_in_perimeter_mode(self):
        """Perimeter mode should have no NaN values (gaps are forward-filled)."""
        da = self.ds['MaskConfidence'].load()
        self.assertFalse(da.isnull().any().item())

    def test_values_in_valid_range(self):
        """All confidence values should be in [0.0, 1.0]."""
        da = self.ds['MaskConfidence'].load()
        self.assertTrue((da >= 0.0).all().item())
        self.assertTrue((da <= 1.0).all().item())

    def test_monotonically_non_decreasing_per_pixel(self):
        """In perimeter (cummax) mode, each pixel should never decrease over time."""
        da = self.ds['MaskConfidence'].load()
        diff = da.diff(dim='time')
        self.assertTrue(
            (diff >= 0).all().item(),
            "Perimeter cummax violated: some pixel decreased over time."
        )

    def test_hour1_has_2x2_fire(self):
        """Hour 1: fire should be 2x2 in center (pixels [4:6, 4:6])."""
        da = self.ds['MaskConfidence'].isel(time=0).load()
        # Center 2x2 should be 1.0
        np.testing.assert_array_equal(da.values[4:6, 4:6], 1.0)
        # Total fire pixels at hour 1 should be 4
        self.assertEqual(int((da > 0).sum()), 4)

    def test_hour2_has_grown_fire(self):
        """Hour 2: fire grows. Should have more fire pixels than hour 1."""
        da_h1 = self.ds['MaskConfidence'].isel(time=0).load()
        da_h2 = self.ds['MaskConfidence'].isel(time=1).load()
        self.assertGreater(int((da_h2 > 0).sum()), int((da_h1 > 0).sum()))

    def test_hour2_subhourly_max_takes_best_confidence(self):
        """
        Hour 2 has two observations: one with mask=13 (conf=0.5) at pixel [3,5],
        and one with mask=10 (conf=1.0) at the same pixel. Max should win.
        """
        da = self.ds['MaskConfidence'].isel(time=1).load()
        self.assertAlmostEqual(float(da.values[3, 5]), 1.0, places=1)

    def test_gap_hour_is_forward_filled(self):
        """
        Hour 3 (index 2) is a gap. In perimeter mode, it should be
        forward-filled from hour 2 (not zero).
        """
        da_h2 = self.ds['MaskConfidence'].isel(time=1).load()
        da_h3 = self.ds['MaskConfidence'].isel(time=2).load()
        np.testing.assert_array_equal(da_h2.values, da_h3.values)

    def test_hour5_unchanged_from_hour4(self):
        """Hour 5 fire is same shape as hour 4 — cummax should be identical."""
        da_h4 = self.ds['MaskConfidence'].isel(time=3).load()
        da_h5 = self.ds['MaskConfidence'].isel(time=4).load()
        np.testing.assert_array_equal(da_h4.values, da_h5.values)


class TestTemporalDownsamplerActiveFire(unittest.TestCase):
    """
    Test aggregate() in active fire mode (is_perimeter=False).

    Same data as above, but without cummax. Gaps should be filled with zeros.
    """

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = TEST_DATA_DIR / "hourly_output_active"
        cls.ds = aggregate(
            goes_save_dir=GOES_SAVE_DIR,
            csv_path=str(CSV_PATH),
            temp_dir=str(cls.temp_dir),
            dates=DATES,
            data_var='MaskConfidence',
            fire_name='test_fire',
            is_perimeter=False,
            verbose=False,
        )

    @classmethod
    def tearDownClass(cls):
        """Clean up intermediate hourly nc files."""
        if cls.temp_dir.exists():
            shutil.rmtree(cls.temp_dir)

    def test_no_nans_in_active_fire_mode(self):
        """Active fire mode should have no NaN values (gaps are zero-filled)."""
        da = self.ds['MaskConfidence'].load()
        self.assertFalse(da.isnull().any().item())

    def test_gap_hour_is_zero_filled(self):
        """Hour 3 (index 2) is a gap. In active fire mode, it should be all zeros."""
        da = self.ds['MaskConfidence'].isel(time=2).load()
        np.testing.assert_array_equal(da.values, 0.0)

    def test_not_monotonically_increasing(self):
        """
        Active fire mode should NOT be cummax'd. Later frames can have fewer
        fire pixels than earlier frames.
        """
        da = self.ds['MaskConfidence'].load()
        # Hour 5 has same fire as hour 4, but hour 3 (gap) has zero.
        # So the sequence is not monotonically non-decreasing.
        sums = da.sum(dim=['y', 'x']).values
        # The gap hour should be 0, breaking monotonicity
        self.assertEqual(float(sums[2]), 0.0)
        self.assertGreater(float(sums[1]), 0.0)


class TestSingleFileHour(unittest.TestCase):
    """
    Test that _open_and_combine_ds handles hours with only a single GOES file.

    When only one file exists for an hour, 't' is a scalar coordinate, causing
    decoded_times to be a scalar Timestamp. This must be handled so
    assign_coords(time=...) doesn't conflict with the 'time' dimension created
    by concat_dim.
    """

    def test_single_file_does_not_raise(self):
        """Opening a single real GOES file should not raise ValueError."""
        from gofer.temporal_downsampler import _open_and_combine_ds

        goes_save_dir = "tests/gofer/data"
        filepath = "temporal/OR_ABI-L2-FDCC-M6_G18_s20250131411181_e20250131413554_c20250131414126.nc"

        ds = _open_and_combine_ds(
            goes_save_dir=goes_save_dir,
            goes_filepaths=[filepath],
        )

        self.assertIn("time", ds.dims)
        self.assertEqual(ds.sizes["time"], 1)
        self.assertIn("time", ds.coords)
        ds.close()


if __name__ == "__main__":
    unittest.main()
