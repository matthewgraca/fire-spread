import unittest
import warnings

warnings.filterwarnings("ignore", message="numpy.ndarray size changed")

import numpy as np
import xarray as xr
import pandas as pd
from gofer.composite import composite


def _make_ortho_ds(values: float, n_times: int = 3) -> xr.Dataset:
    """Create a synthetic orthorectified Dataset with lat/lon coords."""
    times = pd.date_range("2020-09-06 20:00", periods=n_times, freq="h")
    lat = np.linspace(34.5, 34.3, 10)
    lon = np.linspace(-118.1, -117.9, 10)
    data = np.full((n_times, 10, 10), values, dtype=np.float32)

    return xr.Dataset(
        data_vars={
            "MaskConfidence": xr.DataArray(
                data, dims=["time", "latitude", "longitude"]
            ),
        },
        coords={
            "time": times,
            "latitude": lat,
            "longitude": lon,
        },
        attrs={"fire_name": "test_fire"},
    )


class TestComposite(unittest.TestCase):
    """Test that composite merges East and West correctly."""

    @classmethod
    def setUpClass(cls):
        cls.dates = pd.date_range("2020-09-06 20:00", periods=3, freq="h")
        cls.west_ds = _make_ortho_ds(0.6, n_times=3)
        cls.east_ds = _make_ortho_ds(0.4, n_times=3)
        cls.result = composite(
            cls.west_ds, cls.east_ds, cls.dates, data_var="MaskConfidence"
        )

    def test_output_has_no_satellite_dimension(self):
        """Composite should collapse the satellite dimension."""
        self.assertNotIn("satellite", self.result.dims)

    def test_values_in_valid_range(self):
        """All values should be in [0.0, 1.0]."""
        values = self.result["MaskConfidence"].values
        self.assertTrue(np.all(values >= 0.0))
        self.assertTrue(np.all(values <= 1.0))

    def test_output_is_mean_of_east_and_west(self):
        """Composite should be the mean of East (0.4) and West (0.6) = 0.5."""
        values = self.result["MaskConfidence"].values
        np.testing.assert_array_almost_equal(values, 0.5, decimal=5)

    def test_output_lat_lon_match_input(self):
        """Output coordinates should match the input datasets."""
        np.testing.assert_array_equal(
            self.result["latitude"].values,
            self.west_ds["latitude"].values,
        )
        np.testing.assert_array_equal(
            self.result["longitude"].values,
            self.west_ds["longitude"].values,
        )

    def test_output_time_matches_input(self):
        """Output time should match input time."""
        np.testing.assert_array_equal(
            self.result["time"].values,
            self.west_ds["time"].values,
        )


if __name__ == "__main__":
    unittest.main()
