import unittest
import warnings
import tempfile
from pathlib import Path

warnings.filterwarnings("ignore", message="numpy.ndarray size changed")
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import xarray as xr
import pandas as pd
import rioxarray

from gofer.early_perimeter_adjustment import get_scaling_factors, apply_scaling_factors


def _make_synthetic_dem(filepath: str, bbox: tuple, shape: tuple = (20, 20)):
    """Create a small flat DEM GeoTIFF."""
    min_lon, min_lat, max_lon, max_lat = bbox
    lon = np.linspace(min_lon, max_lon, shape[1])
    lat = np.linspace(max_lat, min_lat, shape[0])
    elevation = np.full(shape, 1000.0, dtype=np.float32)

    da = xr.DataArray(
        elevation, dims=["y", "x"], coords={"y": lat, "x": lon}
    )
    da = da.rio.set_spatial_dims(x_dim="x", y_dim="y")
    da = da.rio.write_crs("EPSG:4326")
    da = da.rio.write_nodata(-9999.0)
    da.rio.to_raster(filepath)


BBOX = (-118.1, 34.2, -117.9, 34.4)

GOES_PROJ_ATTRS = {
    "semi_major_axis": 6378137.0,
    "semi_minor_axis": 6356752.31414,
    "perspective_point_height": 35786023.0,
    "longitude_of_projection_origin": -137.0,
}


def _make_growing_fire_ds(n_times: int = 10) -> xr.Dataset:
    """
    Create a synthetic cummax dataset where fire grows over time.
    
    Fire starts as a small patch with low confidence and grows to
    full confidence, simulating a cummax perimeter product.
    """
    x = np.linspace(-0.045, -0.042, 15).astype(np.float64)
    y = np.linspace(0.098, 0.101, 15).astype(np.float64)
    times = pd.date_range("2020-09-06 20:00", periods=n_times, freq="h")

    data = np.zeros((n_times, 15, 15), dtype=np.float32)
    for t in range(n_times):
        # Fire grows and confidence increases over time
        # Max confidence at time t: (t+1) / n_times
        max_conf = (t + 1) / n_times
        radius = min(2 + t, 7)
        center = 7
        data[t, center - radius:center + radius,
             center - radius:center + radius] = max_conf

    # Make it cummax (each frame >= previous)
    for t in range(1, n_times):
        data[t] = np.maximum(data[t], data[t - 1])

    ds = xr.Dataset(
        data_vars={
            "MaskConfidence": xr.DataArray(data, dims=["time", "y", "x"]),
            "goes_imager_projection": xr.DataArray(
                np.int32(0), attrs=GOES_PROJ_ATTRS
            ),
        },
        coords={"x": x, "y": y, "time": times},
    )
    return ds


class TestGetScalingFactors(unittest.TestCase):
    """Test get_scaling_factors produces valid scaling factors."""

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp()
        cls.dem_path = str(Path(cls.tmp_dir) / "test_dem.tif")
        _make_synthetic_dem(cls.dem_path, BBOX, shape=(20, 20))

        cls.ds = _make_growing_fire_ds(n_times=10)
        cls.scaling_factors = get_scaling_factors(
            cls.ds,
            ortho_kwargs={"dem_filepath": cls.dem_path, "bbox": BBOX},
            data_var="MaskConfidence",
            show_progress=False,
        )

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls.tmp_dir)

    def test_scaling_factors_length_matches_time(self):
        """Should produce one scaling factor per timestep."""
        self.assertEqual(len(self.scaling_factors), 10)

    def test_scaling_factors_non_negative(self):
        """All scaling factors should be >= 0."""
        self.assertTrue(np.all(self.scaling_factors >= 0.0))

    def test_no_scaling_factors_between_0_and_tenth(self):
        """
        After cleanup, scaling factors should either be 0 (dropped)
        or >= 1.0 (the reciprocal of values in [0.1, 1.0]).
        Values below 0.1 get set to 0 (reciprocal not computed).
        """
        sf = self.scaling_factors
        # Non-zero values should be >= 1.0 (since they are 1/s where s in [0.1, 1.0])
        nonzero = sf[sf > 0]
        if len(nonzero) > 0:
            self.assertTrue(np.all(nonzero >= 1.0),
                            f"Found scaling factor in invalid range: {nonzero[nonzero < 1.0]}")

    def test_scaling_stops_at_one(self):
        """
        Once the max smoothed confidence reaches 1.0, remaining scaling
        factors should be 1.0 (the reciprocal of 1.0).
        """
        # Find first sf == 1.0; all after should also be 1.0
        ones = np.where(self.scaling_factors == 1.0)[0]
        if len(ones) > 0:
            first_one = ones[0]
            np.testing.assert_array_equal(
                self.scaling_factors[first_one:],
                1.0,
                err_msg="Scaling factors after first 1.0 are not all 1.0",
            )


class TestApplyScalingFactors(unittest.TestCase):
    """Test that apply_scaling_factors produces reasonable output."""

    def test_scaling_does_not_produce_extreme_values(self):
        """
        After applying scaling factors (as multiplication by reciprocal),
        values should not explode beyond a reasonable bound.
        Since max input is 1.0 and max scaling factor is 1/0.1 = 10,
        output should be <= 10.0 at most.
        """
        times = pd.date_range("2020-09-06 20:00", periods=5, freq="h")
        data = np.full((5, 10, 10), 0.5, dtype=np.float32)
        ds = xr.Dataset(
            data_vars={"MaskConfidence": xr.DataArray(
                data, dims=["time", "y", "x"]
            )},
            coords={"time": times},
        )

        # Scaling factors as reciprocals: 1/0.5=2, 1/0.8=1.25, 1.0, 1.0, 1.0
        sf = np.array([2.0, 1.25, 1.0, 1.0, 1.0])
        result = apply_scaling_factors(ds, sf)
        values = result["MaskConfidence"].values

        self.assertTrue(np.all(values <= 10.0),
                        f"Extreme value found: {values.max()}")
        self.assertTrue(np.all(values >= 0.0))

    def test_scaling_factor_of_one_is_identity(self):
        """Scaling factor of 1.0 should leave data unchanged."""
        times = pd.date_range("2020-09-06 20:00", periods=3, freq="h")
        data = np.random.rand(3, 10, 10).astype(np.float32)
        ds = xr.Dataset(
            data_vars={"MaskConfidence": xr.DataArray(
                data, dims=["time", "y", "x"]
            )},
            coords={"time": times},
        )

        sf = np.array([1.0, 1.0, 1.0])
        result = apply_scaling_factors(ds, sf)
        np.testing.assert_array_almost_equal(
            result["MaskConfidence"].values, data, decimal=5
        )

    def test_zero_scaling_factor_zeros_out_frame(self):
        """
        A scaling factor of 0.0 (dropped frame) should zero out that timestep.
        """
        times = pd.date_range("2020-09-06 20:00", periods=3, freq="h")
        data = np.full((3, 10, 10), 0.8, dtype=np.float32)
        ds = xr.Dataset(
            data_vars={"MaskConfidence": xr.DataArray(
                data, dims=["time", "y", "x"]
            )},
            coords={"time": times},
        )

        sf = np.array([0.0, 1.0, 1.0])
        result = apply_scaling_factors(ds, sf)
        # Frame 0 multiplied by 0 should be all zeros
        np.testing.assert_array_equal(
            result["MaskConfidence"].isel(time=0).values, 0.0
        )


if __name__ == "__main__":
    unittest.main()
