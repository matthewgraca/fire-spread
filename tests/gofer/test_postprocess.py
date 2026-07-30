import unittest
import warnings

warnings.filterwarnings("ignore", message="numpy.ndarray size changed")

import numpy as np
import xarray as xr
import pandas as pd
from gofer.postprocess import trim_inactive_timesteps, binarize, round_to


def _make_perimeter_ds(n_times: int = 10) -> xr.Dataset:
    """
    Create a synthetic cummax perimeter dataset.
    
    - First 2 frames: no fire (all zeros)
    - Frames 3-7: fire grows
    - Frames 8-10: fire unchanged (should be trimmed from end)
    """
    times = pd.date_range("2020-09-06 20:00", periods=n_times, freq="h")
    data = np.zeros((n_times, 10, 10), dtype=np.float32)

    # No fire for first 2 frames
    # Fire appears at frame 2 (index 2) and grows
    for t in range(2, 7):
        radius = t - 1
        data[t, 5 - radius:5 + radius, 5 - radius:5 + radius] = 1.0

    # Frames 7-9: same as frame 6 (unchanged)
    for t in range(7, n_times):
        data[t] = data[6]

    return xr.Dataset(
        data_vars={"MaskConfidence": xr.DataArray(
            data, dims=["time", "latitude", "longitude"]
        )},
        coords={
            "time": times,
            "latitude": np.linspace(34.5, 34.4, 10),
            "longitude": np.linspace(-118.1, -118.0, 10),
        },
    )


class TestTrimInactiveTimesteps(unittest.TestCase):
    """Test trim_inactive_timesteps removes leading/trailing inactive frames."""

    @classmethod
    def setUpClass(cls):
        cls.ds = _make_perimeter_ds(n_times=10)

        # trim_inactive_timesteps expects a filepath, so write to a temp file
        # using the same int8 packed encoding the pipeline uses
        import tempfile, os
        cls._tmpdir = tempfile.mkdtemp()
        cls._filepath = os.path.join(cls._tmpdir, 'test_trim.nc')
        encoding = {
            'MaskConfidence': {
                'dtype': 'int8',
                'scale_factor': np.float32(0.01),
                'add_offset': np.float32(0.0),
                '_FillValue': np.int8(-1),
            }
        }
        cls.ds.to_netcdf(cls._filepath, encoding=encoding)

        first_fire, last_change = trim_inactive_timesteps(cls._filepath)
        cls.trimmed = cls.ds.isel(time=slice(first_fire, last_change + 1))

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls._tmpdir, ignore_errors=True)

    def test_first_frame_has_fire(self):
        """After trimming, the first frame should contain fire pixels."""
        first_frame = self.trimmed["MaskConfidence"].isel(time=0)
        self.assertTrue(first_frame.any().item())

    def test_last_frame_differs_from_second_to_last(self):
        """After trimming, the last frame should differ from its predecessor."""
        da = self.trimmed["MaskConfidence"]
        last = da.isel(time=-1).values
        second_to_last = da.isel(time=-2).values
        self.assertFalse(np.array_equal(last, second_to_last))

    def test_leading_empty_frames_removed(self):
        """The original had 2 leading empty frames; they should be gone."""
        # Original first fire at index 2 (2020-09-06 22:00)
        first_time = pd.Timestamp(self.trimmed["time"].values[0])
        expected = pd.Timestamp("2020-09-06 22:00")
        self.assertEqual(first_time, expected)

    def test_trailing_identical_frames_removed(self):
        """
        Original had frames 7-9 identical to frame 6.
        After trim, last frame should be index 6 (the last that changed).
        """
        last_time = pd.Timestamp(self.trimmed["time"].values[-1])
        # Frame 6 = 2020-09-07 02:00 (index 6 from 20:00 start)
        expected = pd.Timestamp("2020-09-07 02:00")
        self.assertEqual(last_time, expected)

    def test_monotonically_non_decreasing_pixel_count(self):
        """After trim, pixel count should be monotonically non-decreasing."""
        da = self.trimmed["MaskConfidence"]
        sums = da.sum(dim=["latitude", "longitude"]).values
        diffs = np.diff(sums)
        self.assertTrue(np.all(diffs >= 0),
                        f"Pixel count decreased: {sums}")


class TestBinarize(unittest.TestCase):
    """Test binarize produces only 0 and 1."""

    def test_output_contains_only_0_and_1(self):
        """After binarize, all values should be exactly 0.0 or 1.0."""
        data = np.array([[[0.1, 0.5, 0.94, 0.95, 0.99, 1.0,
                           0.0, 0.949, 0.951, 0.3]]], dtype=np.float32)
        ds = xr.Dataset(
            data_vars={"MaskConfidence": xr.DataArray(
                data, dims=["time", "latitude", "longitude"]
            )},
            coords={
                "time": [pd.Timestamp("2020-01-01")],
                "latitude": [34.5],
                "longitude": np.arange(10, dtype=np.float64),
            },
        )
        result = binarize(ds, threshold=0.95)
        values = result["MaskConfidence"].values
        unique = np.unique(values)
        np.testing.assert_array_equal(sorted(unique), [0.0, 1.0])

    def test_below_threshold_is_zero(self):
        """Values below threshold should become 0."""
        data = np.array([[[0.94]]], dtype=np.float32)
        ds = xr.Dataset(
            data_vars={"MaskConfidence": xr.DataArray(
                data, dims=["time", "latitude", "longitude"]
            )},
            coords={
                "time": [pd.Timestamp("2020-01-01")],
                "latitude": [34.5],
                "longitude": [-118.0],
            },
        )
        result = binarize(ds, threshold=0.95)
        self.assertEqual(float(result["MaskConfidence"].values[0, 0, 0]), 0.0)

    def test_at_threshold_is_one(self):
        """Values exactly at threshold should become 1."""
        data = np.array([[[0.95]]], dtype=np.float32)
        ds = xr.Dataset(
            data_vars={"MaskConfidence": xr.DataArray(
                data, dims=["time", "latitude", "longitude"]
            )},
            coords={
                "time": [pd.Timestamp("2020-01-01")],
                "latitude": [34.5],
                "longitude": [-118.0],
            },
        )
        result = binarize(ds, threshold=0.95)
        self.assertEqual(float(result["MaskConfidence"].values[0, 0, 0]), 1.0)


class TestRoundTo(unittest.TestCase):
    """Test round_to rounds correctly."""

    def test_rounds_to_2_decimals(self):
        """Values should be rounded to 2 decimal places."""
        data = np.array([[[0.946, 0.944, 0.955]]], dtype=np.float32)
        ds = xr.Dataset(
            data_vars={"MaskConfidence": xr.DataArray(
                data, dims=["time", "latitude", "longitude"]
            )},
            coords={
                "time": [pd.Timestamp("2020-01-01")],
                "latitude": [34.5],
                "longitude": np.arange(3, dtype=np.float64),
            },
        )
        result = round_to(ds, decimals=2)
        values = result["MaskConfidence"].values[0, 0]
        np.testing.assert_array_almost_equal(
            values, [0.95, 0.94, 0.96], decimal=2
        )


if __name__ == "__main__":
    unittest.main()
