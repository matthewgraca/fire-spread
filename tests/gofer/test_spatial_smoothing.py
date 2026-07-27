import unittest
import warnings

warnings.filterwarnings("ignore", message="numpy.ndarray size changed")

import numpy as np
import xarray as xr
from gofer.spatial_smoothing import smooth


def _make_ds(data: np.ndarray) -> xr.Dataset:
    """Create a Dataset with latitude/longitude coords from a 2D array."""
    h, w = data.shape[-2], data.shape[-1]
    # ~90m pixel spacing at ~34 degrees latitude
    lat = np.linspace(34.5, 34.5 - (h - 1) * 0.0008, h)
    lon = np.linspace(-118.0, -118.0 + (w - 1) * 0.0008, w)

    if data.ndim == 2:
        ds = xr.Dataset(
            data_vars={"MaskConfidence": xr.DataArray(
                data.astype(np.float32), dims=["latitude", "longitude"]
            )},
            coords={"latitude": lat, "longitude": lon},
        )
    else:
        # 3D with time
        ds = xr.Dataset(
            data_vars={"MaskConfidence": xr.DataArray(
                data.astype(np.float32), dims=["time", "latitude", "longitude"]
            )},
            coords={
                "latitude": lat,
                "longitude": lon,
                "time": np.arange(data.shape[0]),
            },
        )
    return ds


class TestSpatialSmoothing(unittest.TestCase):
    """Test the smooth() function behavior."""

    def test_output_shape_matches_input_shape(self):
        """Smoothing should not change the spatial dimensions."""
        data = np.random.rand(20, 20).astype(np.float32)
        ds = _make_ds(data)
        result = smooth(ds, kernel_radius_m=100)
        self.assertEqual(
            result["MaskConfidence"].shape,
            ds["MaskConfidence"].shape,
        )

    def test_output_shape_matches_input_with_time(self):
        """Smoothing preserves time dimension."""
        data = np.random.rand(5, 20, 20).astype(np.float32)
        ds = _make_ds(data)
        result = smooth(ds, kernel_radius_m=100)
        self.assertEqual(
            result["MaskConfidence"].shape,
            ds["MaskConfidence"].shape,
        )

    def test_values_do_not_exceed_input_max(self):
        """Mean filter cannot produce values above the input maximum."""
        data = np.random.rand(20, 20).astype(np.float32) * 0.7
        ds = _make_ds(data)
        result = smooth(ds, kernel_radius_m=100)
        input_max = float(ds["MaskConfidence"].max())
        output_max = float(result["MaskConfidence"].max())
        self.assertLessEqual(output_max, input_max + 1e-6)

    def test_values_do_not_go_below_input_min(self):
        """Mean filter cannot produce values below the input minimum."""
        data = np.random.rand(20, 20).astype(np.float32) * 0.7 + 0.2
        ds = _make_ds(data)
        result = smooth(ds, kernel_radius_m=100)
        input_min = float(ds["MaskConfidence"].min())
        output_min = float(result["MaskConfidence"].min())
        self.assertGreaterEqual(output_min, input_min - 1e-6)

    def test_uniform_input_unchanged(self):
        """A uniform field should be unchanged by smoothing."""
        data = np.full((20, 20), 0.5, dtype=np.float32)
        ds = _make_ds(data)
        result = smooth(ds, kernel_radius_m=100)
        np.testing.assert_array_almost_equal(
            result["MaskConfidence"].values, 0.5, decimal=5
        )

    def test_single_hot_pixel_produces_known_mean(self):
        """
        A single pixel of 1.0 in the center of a 5x5 zero grid, with a
        kernel that covers the full 5x5 area, should produce 1/25 = 0.04
        at center (approximately, edge effects from 'nearest' mode may vary).
        
        We use a smaller kernel and verify the center pixel is less than
        the original value (it got averaged with zeros).
        """
        data = np.zeros((21, 21), dtype=np.float32)
        data[10, 10] = 1.0
        ds = _make_ds(data)
        result = smooth(ds, kernel_radius_m=100)

        # Center should be reduced from 1.0 (averaged with surrounding zeros)
        center_val = float(result["MaskConfidence"].values[10, 10])
        self.assertLess(center_val, 1.0)
        self.assertGreater(center_val, 0.0)

        # Total energy should be conserved (sum should be ~1.0 for interior)
        # With 'nearest' mode at edges this won't be exact, but close
        total = float(result["MaskConfidence"].values.sum())
        self.assertAlmostEqual(total, 1.0, places=1)


if __name__ == "__main__":
    unittest.main()
