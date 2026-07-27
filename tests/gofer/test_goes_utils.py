import unittest

import pandas as pd
from gofer.goes_utils import get_satellite


class TestGetSatellite(unittest.TestCase):
    """Tests for get_satellite date/position routing."""

    def test_east_2018_returns_goes16(self):
        result = get_satellite("EAST", pd.Timestamp("2018-01-01"))
        self.assertEqual(result, "noaa-goes16")

    def test_west_2020_returns_goes17(self):
        result = get_satellite("WEST", pd.Timestamp("2020-09-01"))
        self.assertEqual(result, "noaa-goes17")

    def test_west_2024_returns_goes18(self):
        result = get_satellite("WEST", pd.Timestamp("2024-01-01"))
        self.assertEqual(result, "noaa-goes18")

    def test_east_2026_returns_goes19(self):
        result = get_satellite("EAST", pd.Timestamp("2026-01-01"))
        self.assertEqual(result, "noaa-goes19")

    def test_west_overlap_prefers_newest(self):
        """During the GOES-17/18 overlap (mid-2022), prefer GOES-18."""
        result = get_satellite("WEST", pd.Timestamp("2022-10-01"))
        self.assertEqual(result, "noaa-goes18")

    def test_west_before_goes17_raises(self):
        """No WEST satellite available before GOES-17 data starts."""
        with self.assertRaises(ValueError):
            get_satellite("WEST", pd.Timestamp("2017-01-01"))

    def test_invalid_position_raises(self):
        with self.assertRaises(ValueError):
            get_satellite("NORTH", pd.Timestamp("2020-01-01"))

    def test_position_is_case_insensitive(self):
        result = get_satellite("west", pd.Timestamp("2020-09-01"))
        self.assertEqual(result, "noaa-goes17")


if __name__ == "__main__":
    unittest.main()
