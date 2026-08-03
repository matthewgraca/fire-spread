"""
Tests for gofer.metrics — both in-memory and streaming implementations.

Validates that compute_metrics_streaming() produces identical results to the
in-memory fline_r/fspread_mae/fspread_awe/fline_c functions.
"""
import os
import tempfile
import unittest
import warnings

warnings.filterwarnings("ignore", message="numpy.ndarray size changed")

import numpy as np
import pandas as pd
import xarray as xr

from gofer.metrics import (
    _perimeter_mask,
    _fire_line_r_mask,
    _first_fire_timestep,
    fline_r,
    fline_c,
    fspread_mae,
    fspread_awe,
    compute_metrics_streaming,
)


def _make_growing_fire_ds(n_times: int = 20, grid_size: int = 50):
    """
    Create a synthetic cumulative fire perimeter dataset.

    - First 3 frames: no fire
    - Frames 3 onward: fire grows as an expanding square from center
    - Includes both MaskConfidence (binary 0/1) and ActiveFireConfidence
    """
    times = pd.date_range("2021-07-14 12:00", periods=n_times, freq="h")
    mc_data = np.zeros((n_times, grid_size, grid_size), dtype=np.float32)
    afc_data = np.zeros((n_times, grid_size, grid_size), dtype=np.float32)

    center = grid_size // 2
    for t in range(3, n_times):
        radius = min(t - 2, center - 1)
        mc_data[t, center - radius:center + radius + 1,
                   center - radius:center + radius + 1] = 1.0

        # ActiveFireConfidence: only the newly burned ring is "active"
        if t > 3:
            prev_radius = min(t - 3, center - 1)
            # Set confidence on the new ring
            ring = mc_data[t].copy()
            ring[center - prev_radius:center + prev_radius + 1,
                 center - prev_radius:center + prev_radius + 1] = 0.0
            afc_data[t] = ring * 0.8  # 80% confidence on active front
        else:
            afc_data[t] = mc_data[t] * 0.8

    ds = xr.Dataset(
        data_vars={
            "MaskConfidence": xr.DataArray(
                mc_data, dims=["time", "latitude", "longitude"]
            ),
            "ActiveFireConfidence": xr.DataArray(
                afc_data, dims=["time", "latitude", "longitude"]
            ),
        },
        coords={
            "time": times,
            "latitude": np.linspace(40.0, 39.5, grid_size),
            "longitude": np.linspace(-121.5, -121.0, grid_size),
        },
    )
    return ds


def _save_ds_as_h5netcdf(ds: xr.Dataset, path: str):
    """
    Save a dataset in the same format the pipeline produces (int8 encoded).
    """
    import h5netcdf

    n_times = ds.sizes['time']
    lat_vals = ds['latitude'].values
    lon_vals = ds['longitude'].values
    time_vals = ds['time'].values

    time_minutes = (
        (time_vals - np.datetime64('1970-01-01T00:00:00', 'ns'))
        // np.timedelta64(1, 'm')
    )

    with h5netcdf.File(path, 'w') as f:
        f.dimensions = {
            'time': n_times,
            'latitude': len(lat_vals),
            'longitude': len(lon_vals),
        }

        time_var = f.create_variable('time', ('time',), data=time_minutes.astype('int32'))
        time_var.attrs['units'] = 'minutes since 1970-01-01'
        time_var.attrs['calendar'] = 'proleptic_gregorian'

        f.create_variable('latitude', ('latitude',), data=lat_vals.astype('float32'))
        f.create_variable('longitude', ('longitude',), data=lon_vals.astype('float32'))

        # Encode MaskConfidence as int8 (value / 0.01)
        mc_raw = np.clip(ds['MaskConfidence'].values / 0.01, 0, 100).astype(np.int8)
        mc_var = f.create_variable(
            'MaskConfidence', ('time', 'latitude', 'longitude'),
            dtype='int8', fillvalue=np.int8(-1), data=mc_raw,
        )
        mc_var.attrs['scale_factor'] = np.float32(0.01)
        mc_var.attrs['add_offset'] = np.float32(0.0)

        # Encode ActiveFireConfidence as int8
        afc_raw = np.clip(ds['ActiveFireConfidence'].values / 0.01, 0, 100).astype(np.int8)
        afc_var = f.create_variable(
            'ActiveFireConfidence', ('time', 'latitude', 'longitude'),
            dtype='int8', fillvalue=np.int8(-1), data=afc_raw,
        )
        afc_var.attrs['scale_factor'] = np.float32(0.01)
        afc_var.attrs['add_offset'] = np.float32(0.0)


class TestPerimeterMask(unittest.TestCase):
    """Test the boundary extraction helper."""

    def test_empty_mask_returns_empty(self):
        mask = np.zeros((10, 10), dtype=bool)
        result = _perimeter_mask(mask)
        self.assertFalse(result.any())

    def test_single_pixel_is_its_own_perimeter(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[5, 5] = True
        result = _perimeter_mask(mask)
        self.assertTrue(result[5, 5])
        self.assertEqual(result.sum(), 1)

    def test_filled_square_perimeter(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[3:7, 3:7] = True  # 4x4 square
        result = _perimeter_mask(mask)
        # Interior pixels (4,4), (4,5), (5,4), (5,5) should NOT be in perimeter
        self.assertFalse(result[4, 4])
        self.assertFalse(result[5, 5])
        # Edge pixels should be in perimeter
        self.assertTrue(result[3, 3])
        self.assertTrue(result[6, 6])


class TestFirstFireTimestep(unittest.TestCase):
    """Test finding the first fire timestep."""

    def test_fire_starts_at_index_3(self):
        data = np.zeros((10, 5, 5), dtype=bool)
        data[3, 2, 2] = True
        self.assertEqual(_first_fire_timestep(data), 3)

    def test_all_empty_returns_0(self):
        data = np.zeros((10, 5, 5), dtype=bool)
        self.assertEqual(_first_fire_timestep(data), 0)


class TestFlineR(unittest.TestCase):
    """Test retrospective fire line length."""

    def test_no_growth_yields_zero(self):
        """If the fire doesn't grow between timesteps, fline_r should be 0."""
        ds = _make_growing_fire_ds(n_times=10)
        # Make all timesteps identical (no growth after t=3)
        for t in range(4, 10):
            ds['MaskConfidence'].values[t] = ds['MaskConfidence'].values[3]

        result = fline_r(ds)
        # After the initial appearance, no growth → fline_r = 0
        self.assertEqual(result.values[4:].sum(), 0.0)

    def test_growing_fire_has_positive_fline(self):
        """A growing fire should have positive fline_r values."""
        ds = _make_growing_fire_ds()
        result = fline_r(ds)
        # There should be positive values during the growth period
        self.assertTrue((result.values > 0).any())

    def test_last_timestep_is_zero(self):
        """fline_r at the last timestep should always be 0."""
        ds = _make_growing_fire_ds()
        result = fline_r(ds)
        self.assertEqual(result.values[-1], 0.0)


class TestStreamingMatchesInMemory(unittest.TestCase):
    """
    Core correctness test: streaming metrics must produce identical results
    to the in-memory implementations when both operate on the same data.

    Both reference and streaming read from the same int8-encoded file to
    eliminate quantization differences.
    """

    @classmethod
    def setUpClass(cls):
        """Create test dataset, save in pipeline format, then compute both."""
        cls.ds = _make_growing_fire_ds(n_times=20, grid_size=50)

        # Save to temp file in pipeline format (int8 encoded)
        cls.tmpfile = tempfile.NamedTemporaryFile(suffix='.nc', delete=False)
        cls.tmpfile.close()
        _save_ds_as_h5netcdf(cls.ds, cls.tmpfile.name)

        # Compute in-memory reference FROM the encoded file (same data path)
        ref_ds = xr.open_dataset(cls.tmpfile.name)
        ref_ds_loaded = ref_ds.load()
        ref_ds.close()

        cls.ref_fline_r = fline_r(ref_ds_loaded)
        cls.ref_mae = fspread_mae(ref_ds_loaded)
        cls.ref_awe = fspread_awe(ref_ds_loaded, fline_r_da=cls.ref_fline_r)
        cls.ref_fline_c = fline_c(
            ref_ds_loaded, confidence_var='ActiveFireConfidence',
            confidence_threshold=0.05,
        )
        del ref_ds_loaded

        # Compute streaming
        cls.streaming_ds = compute_metrics_streaming(
            cls.tmpfile.name, batch_size=5,  # small batch to test boundary handling
        )

    @classmethod
    def tearDownClass(cls):
        os.unlink(cls.tmpfile.name)

    def test_fline_r_matches(self):
        """Streaming fline_r should match in-memory fline_r."""
        np.testing.assert_allclose(
            self.streaming_ds['fline_r'].values,
            self.ref_fline_r.values,
            rtol=1e-5,
            err_msg="fline_r mismatch between streaming and in-memory",
        )

    def test_fspread_mae_matches(self):
        """Streaming fspread_mae should match in-memory fspread_mae."""
        np.testing.assert_allclose(
            self.streaming_ds['fspread_mae'].values,
            self.ref_mae.values,
            rtol=1e-5,
            err_msg="fspread_mae mismatch between streaming and in-memory",
        )

    def test_fspread_awe_matches(self):
        """Streaming fspread_awe should match in-memory fspread_awe."""
        np.testing.assert_allclose(
            self.streaming_ds['fspread_awe'].values,
            self.ref_awe.values,
            rtol=1e-5,
            err_msg="fspread_awe mismatch between streaming and in-memory",
        )

    def test_fline_c_matches(self):
        """Streaming fline_c should match in-memory fline_c."""
        self.assertIn('fline_c', self.streaming_ds)
        np.testing.assert_allclose(
            self.streaming_ds['fline_c'].values,
            self.ref_fline_c.values,
            rtol=1e-5,
            err_msg="fline_c mismatch between streaming and in-memory",
        )

    def test_time_coords_match(self):
        """Time coordinates should be identical."""
        np.testing.assert_array_equal(
            self.streaming_ds['fline_r'].coords['time'].values,
            self.ref_fline_r.coords['time'].values,
        )

    def test_time_mid_coords_match(self):
        """time_mid coordinates should be identical."""
        np.testing.assert_array_equal(
            self.streaming_ds['fspread_mae'].coords['time_mid'].values,
            self.ref_mae.coords['time_mid'].values,
        )


class TestStreamingSmallBatch(unittest.TestCase):
    """Test streaming with batch_size=1 (maximum I/O, tests edge cases)."""

    @classmethod
    def setUpClass(cls):
        cls.ds = _make_growing_fire_ds(n_times=10, grid_size=30)

        cls.tmpfile = tempfile.NamedTemporaryFile(suffix='.nc', delete=False)
        cls.tmpfile.close()
        _save_ds_as_h5netcdf(cls.ds, cls.tmpfile.name)

        # Reference from the same encoded file
        ref_ds = xr.open_dataset(cls.tmpfile.name).load()
        cls.ref_fline_r = fline_r(ref_ds)
        cls.ref_mae = fspread_mae(ref_ds)
        del ref_ds

        cls.streaming_ds = compute_metrics_streaming(
            cls.tmpfile.name, batch_size=1,
        )

    @classmethod
    def tearDownClass(cls):
        os.unlink(cls.tmpfile.name)

    def test_fline_r_matches_batch1(self):
        """batch_size=1 should still produce correct fline_r."""
        np.testing.assert_allclose(
            self.streaming_ds['fline_r'].values,
            self.ref_fline_r.values,
            rtol=1e-5,
        )

    def test_fspread_mae_matches_batch1(self):
        """batch_size=1 should still produce correct fspread_mae."""
        np.testing.assert_allclose(
            self.streaming_ds['fspread_mae'].values,
            self.ref_mae.values,
            rtol=1e-5,
        )


class TestStreamingNoConfidence(unittest.TestCase):
    """Test streaming when ActiveFireConfidence is not in the file."""

    @classmethod
    def setUpClass(cls):
        cls.ds = _make_growing_fire_ds(n_times=10, grid_size=30)
        # Remove confidence variable
        ds_no_afc = cls.ds.drop_vars('ActiveFireConfidence')

        cls.tmpfile = tempfile.NamedTemporaryFile(suffix='.nc', delete=False)
        cls.tmpfile.close()

        # Save without AFC
        import h5netcdf
        n_times = ds_no_afc.sizes['time']
        lat_vals = ds_no_afc['latitude'].values
        lon_vals = ds_no_afc['longitude'].values
        time_vals = ds_no_afc['time'].values
        time_minutes = (
            (time_vals - np.datetime64('1970-01-01T00:00:00', 'ns'))
            // np.timedelta64(1, 'm')
        )

        with h5netcdf.File(cls.tmpfile.name, 'w') as f:
            f.dimensions = {
                'time': n_times,
                'latitude': len(lat_vals),
                'longitude': len(lon_vals),
            }
            time_var = f.create_variable('time', ('time',), data=time_minutes.astype('int32'))
            time_var.attrs['units'] = 'minutes since 1970-01-01'
            time_var.attrs['calendar'] = 'proleptic_gregorian'
            f.create_variable('latitude', ('latitude',), data=lat_vals.astype('float32'))
            f.create_variable('longitude', ('longitude',), data=lon_vals.astype('float32'))

            mc_raw = np.clip(ds_no_afc['MaskConfidence'].values / 0.01, 0, 100).astype(np.int8)
            mc_var = f.create_variable(
                'MaskConfidence', ('time', 'latitude', 'longitude'),
                dtype='int8', fillvalue=np.int8(-1), data=mc_raw,
            )
            mc_var.attrs['scale_factor'] = np.float32(0.01)
            mc_var.attrs['add_offset'] = np.float32(0.0)

        cls.streaming_ds = compute_metrics_streaming(cls.tmpfile.name)

    @classmethod
    def tearDownClass(cls):
        os.unlink(cls.tmpfile.name)

    def test_no_fline_c_when_no_confidence(self):
        """fline_c should not be in output if confidence data is missing."""
        self.assertNotIn('fline_c', self.streaming_ds)

    def test_other_metrics_still_computed(self):
        """fline_r, fspread_mae, fspread_awe should still be present."""
        self.assertIn('fline_r', self.streaming_ds)
        self.assertIn('fspread_mae', self.streaming_ds)
        self.assertIn('fspread_awe', self.streaming_ds)


if __name__ == '__main__':
    unittest.main()
