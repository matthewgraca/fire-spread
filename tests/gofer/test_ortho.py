import unittest
import warnings
import tempfile
from pathlib import Path
import numpy as np
import xarray as xr
import rioxarray

from gofer.ortho import make_ortho_map, apply_ortho_map

# NOTE: you'll see NumPy shape DeprecationWarning -- that's upstream from rioxarray/rasterio
# NOTE: you'll see Error in sys.excepthook -- that's from dask + unittest + Python 3.14
#   as far as I know, these are cosmetic.

# --- Test fixtures ---

# Small bounding box over Southern California (near Bobcat fire)
BBOX = (-118.1, 34.2, -117.9, 34.4)

# GOES-West projection parameters
GOES_PROJ_ATTRS = {
    "semi_major_axis": 6378137.0,
    "semi_minor_axis": 6356752.31414,
    "perspective_point_height": 35786023.0,
    "longitude_of_projection_origin": -137.0,
}


def _make_synthetic_dem(filepath: str, bbox: tuple, shape: tuple = (20, 20)):
    """
    Create a small synthetic DEM GeoTIFF in EPSG:4326.
    Elevation varies linearly from 500m to 2000m across the grid.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    lon = np.linspace(min_lon, max_lon, shape[1])
    lat = np.linspace(max_lat, min_lat, shape[0])  # descending for raster

    # Linear elevation gradient
    elevation = np.linspace(500, 2000, shape[0])[:, np.newaxis] * np.ones(shape[1])

    da = xr.DataArray(
        elevation.astype(np.float32),
        dims=["y", "x"],
        coords={"y": lat, "x": lon},
    )
    da = da.rio.set_spatial_dims(x_dim="x", y_dim="y")
    da = da.rio.write_crs("EPSG:4326")
    da = da.rio.write_nodata(-9999.0)
    da.rio.to_raster(filepath)


def _make_synthetic_goes_ds(shape: tuple = (10, 10)) -> xr.Dataset:
    """
    Create a synthetic GOES-like Dataset with ABI fixed-grid x/y coordinates
    and a MaskConfidence data variable.

    The x/y ranges are chosen to cover the same region as the DEM bbox
    when viewed from GOES-West at -137 deg.
    """
    # ABI scan angles roughly corresponding to Southern California from GOES-West
    # These are approximate values for the test bbox
    x = np.linspace(-0.045, -0.042, shape[1]).astype(np.float64)
    y = np.linspace(0.098, 0.101, shape[0]).astype(np.float64)

    # Uniform confidence = 0.8 everywhere
    data = np.full(shape, 0.8, dtype=np.float32)

    ds = xr.Dataset(
        data_vars={
            "MaskConfidence": xr.DataArray(data, dims=["y", "x"]),
            "goes_imager_projection": xr.DataArray(
                np.int32(0), attrs=GOES_PROJ_ATTRS
            ),
        },
        coords={
            "x": x,
            "y": y,
        },
    )
    return ds


class TestMakeOrthoMap(unittest.TestCase):
    """Test make_ortho_map produces a valid orthorectification map."""

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp()
        cls.dem_path = str(Path(cls.tmp_dir) / "test_dem.tif")
        _make_synthetic_dem(cls.dem_path, BBOX, shape=(20, 20))

        cls.goes_ds = _make_synthetic_goes_ds(shape=(10, 10))
        cls.ortho_map = make_ortho_map(
            cls.goes_ds,
            dem_filepath=cls.dem_path,
            bbox=BBOX,
            parallax_adjustment_factor=0.85,
            kernel_radius_m=1700,
            include_fixed_grid_diagnostics=True,
        )

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls.tmp_dir)

    def test_output_grid_matches_dem_coordinates(self):
        """Ortho map lat/lon should match the DEM's coordinate arrays."""
        dem_da = rioxarray.open_rasterio(self.dem_path, masked=True).squeeze("band", drop=True)
        dem_clipped = dem_da.rio.clip_box(*BBOX, crs="EPSG:4326")

        np.testing.assert_array_almost_equal(
            self.ortho_map["longitude"].values,
            dem_clipped["x"].values,
            decimal=6,
        )
        np.testing.assert_array_almost_equal(
            self.ortho_map["latitude"].values,
            dem_clipped["y"].values,
            decimal=6,
        )

    def test_scan_angles_within_valid_abi_range(self):
        """
        ABI scan angles should be within the physical range of the instrument.
        x: roughly [-0.15, 0.15] rad, y: roughly [-0.15, 0.15] rad.
        """
        abi_x = self.ortho_map["dem_px_angle_x"].values
        abi_y = self.ortho_map["dem_px_angle_y"].values

        self.assertTrue(np.all(np.abs(abi_x) < 0.15),
                        f"ABI x scan angles out of range: [{abi_x.min()}, {abi_x.max()}]")
        self.assertTrue(np.all(np.abs(abi_y) < 0.15),
                        f"ABI y scan angles out of range: [{abi_y.min()}, {abi_y.max()}]")

    def test_no_nan_in_scan_angles(self):
        """Scan angles should have no NaN when DEM has no NaN elevation."""
        abi_x = self.ortho_map["dem_px_angle_x"].values
        abi_y = self.ortho_map["dem_px_angle_y"].values

        self.assertFalse(np.any(np.isnan(abi_x)), "NaN found in dem_px_angle_x")
        self.assertFalse(np.any(np.isnan(abi_y)), "NaN found in dem_px_angle_y")

    def test_output_shape_matches_dem_grid(self):
        """Ortho map spatial dims should be (DEM_lat, DEM_lon), not (GOES_y, GOES_x)."""
        dem_da = rioxarray.open_rasterio(self.dem_path, masked=True).squeeze("band", drop=True)
        dem_clipped = dem_da.rio.clip_box(*BBOX, crs="EPSG:4326")

        self.assertEqual(self.ortho_map.sizes["latitude"], len(dem_clipped["y"]))
        self.assertEqual(self.ortho_map.sizes["longitude"], len(dem_clipped["x"]))

        # Should NOT match GOES dims
        self.assertNotEqual(self.ortho_map.sizes["latitude"], self.goes_ds.sizes["y"])
        self.assertNotEqual(self.ortho_map.sizes["longitude"], self.goes_ds.sizes["x"])

    def test_has_fixed_grid_diagnostics(self):
        """When requested, ortho map should contain diagnostic variables."""
        self.assertIn("abi_fixed_grid_x", self.ortho_map)
        self.assertIn("abi_fixed_grid_y", self.ortho_map)
        self.assertIn("zone_labels", self.ortho_map)


class TestMakeOrthoMapIdentityParallax(unittest.TestCase):
    """Test that parallax_adjustment_factor=0 still produces valid output."""

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp()
        cls.dem_path = str(Path(cls.tmp_dir) / "test_dem.tif")
        _make_synthetic_dem(cls.dem_path, BBOX, shape=(20, 20))

        cls.goes_ds = _make_synthetic_goes_ds(shape=(10, 10))
        cls.ortho_map = make_ortho_map(
            cls.goes_ds,
            dem_filepath=cls.dem_path,
            bbox=BBOX,
            parallax_adjustment_factor=0.0,
            kernel_radius_m=1700,
            include_fixed_grid_diagnostics=False,
        )

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls.tmp_dir)

    def test_zero_parallax_produces_valid_scan_angles(self):
        """With factor=0, scan angles should still be finite and in range."""
        abi_x = self.ortho_map["dem_px_angle_x"].values
        abi_y = self.ortho_map["dem_px_angle_y"].values

        self.assertTrue(np.all(np.isfinite(abi_x)))
        self.assertTrue(np.all(np.isfinite(abi_y)))
        self.assertTrue(np.all(np.abs(abi_x) < 0.15))
        self.assertTrue(np.all(np.abs(abi_y) < 0.15))

    def test_zero_parallax_ignores_elevation(self):
        """
        With factor=0, the scan angles should be independent of elevation
        (only lon/lat matter). Verify by comparing to a flat DEM.
        """
        flat_dem_path = str(Path(self.tmp_dir) / "flat_dem.tif")
        _make_synthetic_dem(flat_dem_path, BBOX, shape=(20, 20))

        # Overwrite with flat elevation
        dem_da = rioxarray.open_rasterio(flat_dem_path, masked=True).squeeze("band", drop=True)
        flat = xr.zeros_like(dem_da)
        flat = flat.rio.set_spatial_dims(x_dim="x", y_dim="y")
        flat = flat.rio.write_crs("EPSG:4326")
        flat = flat.rio.write_nodata(-9999.0)
        flat.rio.to_raster(flat_dem_path)

        ortho_flat = make_ortho_map(
            self.goes_ds,
            dem_filepath=flat_dem_path,
            bbox=BBOX,
            parallax_adjustment_factor=0.0,
            kernel_radius_m=1700,
            include_fixed_grid_diagnostics=False,
        )

        # With factor=0, elevation doesn't matter — both should be identical
        np.testing.assert_array_almost_equal(
            self.ortho_map["dem_px_angle_x"].values,
            ortho_flat["dem_px_angle_x"].values,
            decimal=10,
        )
        np.testing.assert_array_almost_equal(
            self.ortho_map["dem_px_angle_y"].values,
            ortho_flat["dem_px_angle_y"].values,
            decimal=10,
        )


class TestApplyOrthoMap(unittest.TestCase):
    """Test that apply_ortho_map produces correctly shaped output."""

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp()
        cls.dem_path = str(Path(cls.tmp_dir) / "test_dem.tif")
        _make_synthetic_dem(cls.dem_path, BBOX, shape=(20, 20))

        cls.goes_ds = _make_synthetic_goes_ds(shape=(10, 10))
        cls.ortho_map = make_ortho_map(
            cls.goes_ds,
            dem_filepath=cls.dem_path,
            bbox=BBOX,
            parallax_adjustment_factor=0.85,
            kernel_radius_m=1700,
            include_fixed_grid_diagnostics=True,
        )
        cls.ortho_ds = apply_ortho_map(
            cls.goes_ds,
            cls.ortho_map,
            data_var="MaskConfidence",
        )

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls.tmp_dir)

    def test_output_has_dem_spatial_dims(self):
        """Output should have latitude/longitude dims matching the DEM grid."""
        self.assertIn("latitude", self.ortho_ds.dims)
        self.assertIn("longitude", self.ortho_ds.dims)
        self.assertNotIn("y", self.ortho_ds.dims)
        self.assertNotIn("x", self.ortho_ds.dims)

    def test_output_shape_is_dem_grid(self):
        """Output spatial shape should match the ortho map (DEM grid)."""
        self.assertEqual(
            self.ortho_ds.sizes["latitude"],
            self.ortho_map.sizes["latitude"],
        )
        self.assertEqual(
            self.ortho_ds.sizes["longitude"],
            self.ortho_map.sizes["longitude"],
        )

    def test_output_contains_data_variable(self):
        """The requested data variable should be in the output."""
        self.assertIn("MaskConfidence", self.ortho_ds.data_vars)

    def test_output_values_come_from_source(self):
        """
        Since the source has uniform 0.8 everywhere, the ortho output
        should also be 0.8 everywhere (nearest-neighbor from uniform source).
        """
        values = self.ortho_ds["MaskConfidence"].values
        np.testing.assert_array_almost_equal(values, 0.8, decimal=5)


if __name__ == "__main__":
    unittest.main()
