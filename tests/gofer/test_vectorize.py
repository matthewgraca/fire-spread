import unittest
import warnings

warnings.filterwarnings("ignore", message="numpy.ndarray size changed")
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import xarray as xr
import pandas as pd
from shapely.validation import explain_validity
from gofer.vectorize import raster_to_polygon


def _make_binary_ds(n_times: int = 3) -> xr.Dataset:
    """
    Create a synthetic binary fire perimeter dataset.
    Fire is a growing square in the center.
    """
    lat = np.linspace(34.5, 34.4, 20)
    lon = np.linspace(-118.1, -118.0, 20)
    times = pd.date_range("2020-09-06 20:00", periods=n_times, freq="h")

    data = np.zeros((n_times, 20, 20), dtype=np.float32)
    for t in range(n_times):
        radius = 2 + t * 2
        center = 10
        r_lo = max(center - radius, 0)
        r_hi = min(center + radius, 20)
        data[t, r_lo:r_hi, r_lo:r_hi] = 1.0

    return xr.Dataset(
        data_vars={"MaskConfidence": xr.DataArray(
            data, dims=["time", "latitude", "longitude"]
        )},
        coords={
            "time": times,
            "latitude": lat,
            "longitude": lon,
        },
    )


def _make_single_frame_ds() -> xr.Dataset:
    """Create a single-frame binary dataset with a fire square."""
    lat = np.linspace(34.5, 34.4, 20)
    lon = np.linspace(-118.1, -118.0, 20)

    data = np.zeros((20, 20), dtype=np.float32)
    data[5:15, 5:15] = 1.0

    return xr.Dataset(
        data_vars={"MaskConfidence": xr.DataArray(
            data, dims=["latitude", "longitude"]
        )},
        coords={"latitude": lat, "longitude": lon},
    )


class TestVectorizeMultiTime(unittest.TestCase):
    """Test raster_to_polygon with multi-timestep data."""

    @classmethod
    def setUpClass(cls):
        cls.ds = _make_binary_ds(n_times=4)
        cls.gdf = raster_to_polygon(cls.ds)

    def test_output_has_one_row_per_timestep(self):
        """Should produce one polygon per timestep."""
        self.assertEqual(len(self.gdf), 4)

    def test_output_has_time_column(self):
        """Multi-timestep output should have a time column."""
        self.assertIn("time", self.gdf.columns)

    def test_polygon_area_is_positive(self):
        """Every polygon should have area > 0."""
        for idx, row in self.gdf.iterrows():
            self.assertGreater(row.geometry.area, 0,
                               f"Polygon at index {idx} has zero area")

    def test_polygons_are_valid(self):
        """Every polygon should be geometrically valid (no self-intersections)."""
        for idx, row in self.gdf.iterrows():
            self.assertTrue(
                row.geometry.is_valid,
                f"Invalid polygon at index {idx}: {explain_validity(row.geometry)}"
            )

    def test_polygon_crs_is_epsg4326(self):
        """Output CRS should be EPSG:4326."""
        self.assertEqual(self.gdf.crs.to_epsg(), 4326)

    def test_polygon_bounds_within_dataset_extent(self):
        """All polygon bounds should be within the dataset's lat/lon extent."""
        lon_min = float(self.ds.longitude.min())
        lon_max = float(self.ds.longitude.max())
        lat_min = float(self.ds.latitude.min())
        lat_max = float(self.ds.latitude.max())

        # Allow half-pixel buffer since polygons extend to pixel edges
        dlat = abs(float(self.ds.latitude[1] - self.ds.latitude[0]))
        dlon = abs(float(self.ds.longitude[1] - self.ds.longitude[0]))

        for idx, row in self.gdf.iterrows():
            bounds = row.geometry.bounds  # (minx, miny, maxx, maxy)
            self.assertGreaterEqual(bounds[0], lon_min - dlon,
                                    f"Polygon {idx} extends west of dataset")
            self.assertGreaterEqual(bounds[1], lat_min - dlat,
                                    f"Polygon {idx} extends south of dataset")
            self.assertLessEqual(bounds[2], lon_max + dlon,
                                 f"Polygon {idx} extends east of dataset")
            self.assertLessEqual(bounds[3], lat_max + dlat,
                                 f"Polygon {idx} extends north of dataset")

    def test_later_polygons_are_larger(self):
        """Since fire grows, later polygons should have >= area of earlier ones."""
        areas = [row.geometry.area for _, row in self.gdf.iterrows()]
        for i in range(1, len(areas)):
            self.assertGreaterEqual(areas[i], areas[i - 1])


class TestVectorizeSingleFrame(unittest.TestCase):
    """Test raster_to_polygon with a single frame (no time dimension)."""

    @classmethod
    def setUpClass(cls):
        cls.ds = _make_single_frame_ds()
        cls.gdf = raster_to_polygon(cls.ds)

    def test_output_has_one_row(self):
        """Single frame should produce one polygon."""
        self.assertEqual(len(self.gdf), 1)

    def test_no_time_column(self):
        """Single frame output should not have a time column."""
        self.assertNotIn("time", self.gdf.columns)

    def test_polygon_is_valid(self):
        """Polygon should be geometrically valid."""
        geom = self.gdf.iloc[0].geometry
        self.assertTrue(geom.is_valid, explain_validity(geom))

    def test_polygon_crs_is_epsg4326(self):
        """Output CRS should be EPSG:4326."""
        self.assertEqual(self.gdf.crs.to_epsg(), 4326)

    def test_polygon_area_positive(self):
        """Polygon should have area > 0."""
        self.assertGreater(self.gdf.iloc[0].geometry.area, 0)


class TestVectorizeEmptyFrame(unittest.TestCase):
    """Test raster_to_polygon when there's no fire."""

    def test_empty_frame_returns_empty_gdf(self):
        """A frame with all zeros should return an empty GeoDataFrame."""
        lat = np.linspace(34.5, 34.4, 10)
        lon = np.linspace(-118.1, -118.0, 10)
        data = np.zeros((10, 10), dtype=np.float32)

        ds = xr.Dataset(
            data_vars={"MaskConfidence": xr.DataArray(
                data, dims=["latitude", "longitude"]
            )},
            coords={"latitude": lat, "longitude": lon},
        )
        gdf = raster_to_polygon(ds)
        self.assertEqual(len(gdf), 0)


if __name__ == "__main__":
    unittest.main()
