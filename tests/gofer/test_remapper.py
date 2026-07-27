import unittest
import numpy as np
import xarray as xr
from gofer.remapper import map_fdc_mask_to_confidence 


DQF_ATTRS = {
    "flag_values": [0, 1, 2, 3, 4, 5],
    "flag_meanings": (
        "good_quality_fire_pixel_qf "
        "good_quality_fire_free_land_pixel_qf "
        "invalid_due_to_opaque_cloud_pixel_qf "
        "invalid_due_to_surface_type_or_sunglint_or_LZA_threshold_exceeded_or_off_earth_or_missing_input_data_qf "
        "invalid_due_to_bad_input_data_qf "
        "invalid_due_to_algorithm_failure_qf"
    ),
}


def _make_dataset(mask: np.ndarray, dqf: np.ndarray) -> xr.Dataset:
    """Helper to build a minimal GOES-like Dataset from 2D arrays."""
    return xr.Dataset({
        "Mask": xr.DataArray(mask, dims=["y", "x"]),
        "DQF": xr.DataArray(dqf, dims=["y", "x"], attrs=DQF_ATTRS),
    })


class TestRemapperConfidenceMapping(unittest.TestCase):
    """
    Test that mask codes are correctly mapped to confidence values
    when all pixels have good DQF (DQF=0).
    """

    @classmethod
    def setUpClass(cls):
        # 10x10 grid with all known fire mask codes placed deliberately.
        # Row 0: processed (10, 30)
        # Row 1: saturated (11, 31)
        # Row 2: cloud contaminated (12, 32)
        # Row 3: high probability (13, 33)
        # Row 4: medium probability (14, 34)
        # Row 5: low probability (15, 35)
        # Row 6-7: non-fire codes (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
        # Row 8-9: out-of-range codes (36, 40, 50, 100, 255, etc.)
        mask = np.zeros((10, 10), dtype=np.int16)

        # Row 0: processed
        mask[0, :5] = 10
        mask[0, 5:] = 30

        # Row 1: saturated
        mask[1, :5] = 11
        mask[1, 5:] = 31

        # Row 2: cloud contaminated
        mask[2, :5] = 12
        mask[2, 5:] = 32

        # Row 3: high probability
        mask[3, :5] = 13
        mask[3, 5:] = 33

        # Row 4: medium probability
        mask[4, :5] = 14
        mask[4, 5:] = 34

        # Row 5: low probability
        mask[5, :5] = 15
        mask[5, 5:] = 35

        # Row 6-7: non-fire codes
        mask[6, :] = np.arange(0, 10)
        mask[7, :] = np.arange(0, 10)

        # Row 8-9: out-of-range codes
        mask[8, :] = [36, 37, 38, 39, 40, 50, 60, 100, 200, 255]
        mask[9, :] = [36, 37, 38, 39, 40, 50, 60, 100, 200, 255]

        # All good quality
        dqf = np.zeros((10, 10), dtype=np.int16)

        cls.ds = _make_dataset(mask, dqf)
        cls.result = map_fdc_mask_to_confidence(cls.ds)
        cls.conf = cls.result["MaskConfidence"].values

    def test_output_has_mask_confidence(self):
        self.assertIn("MaskConfidence", self.result.data_vars)

    def test_output_drops_mask_and_dqf(self):
        self.assertNotIn("Mask", self.result.data_vars)
        self.assertNotIn("DQF", self.result.data_vars)

    def test_processed_maps_to_1_0(self):
        np.testing.assert_array_equal(self.conf[0, :], 1.0)

    def test_saturated_maps_to_1_0(self):
        np.testing.assert_array_equal(self.conf[1, :], 1.0)

    def test_cloud_contaminated_maps_to_0_8(self):
        np.testing.assert_array_almost_equal(self.conf[2, :], 0.8)

    def test_high_probability_maps_to_0_5(self):
        np.testing.assert_array_almost_equal(self.conf[3, :], 0.5)

    def test_medium_probability_maps_to_0_3(self):
        np.testing.assert_array_almost_equal(self.conf[4, :], 0.3)

    def test_low_probability_maps_to_0_1(self):
        np.testing.assert_array_almost_equal(self.conf[5, :], 0.1)

    def test_non_fire_codes_map_to_0(self):
        np.testing.assert_array_equal(self.conf[6, :], 0.0)
        np.testing.assert_array_equal(self.conf[7, :], 0.0)

    def test_out_of_range_codes_map_to_0(self):
        np.testing.assert_array_equal(self.conf[8, :], 0.0)
        np.testing.assert_array_equal(self.conf[9, :], 0.0)

    def test_all_values_in_valid_range(self):
        self.assertTrue((self.conf >= 0.0).all())
        self.assertTrue((self.conf <= 1.0).all())


class TestRemapperDQFFiltering(unittest.TestCase):
    """
    Test that pixels with bad DQF values are zeroed out regardless
    of their mask code.
    """

    @classmethod
    def setUpClass(cls):
        # 10x10 grid where all pixels have fire mask code 10 (highest confidence)
        # but DQF varies across rows to test filtering.
        mask = np.full((10, 10), 10, dtype=np.int16)

        dqf = np.zeros((10, 10), dtype=np.int16)
        # Row 0-1: DQF=0 (good quality fire) -> should keep confidence
        dqf[0, :] = 0
        dqf[1, :] = 0
        # Row 2-3: DQF=1 (good quality fire-free land) -> should keep confidence
        dqf[2, :] = 1
        dqf[3, :] = 1
        # Row 4-5: DQF=2 (invalid - opaque cloud) -> should zero out
        dqf[4, :] = 2
        dqf[5, :] = 2
        # Row 6-7: DQF=3 (invalid - surface type/sunglint/etc) -> should zero out
        dqf[6, :] = 3
        dqf[7, :] = 3
        # Row 8: DQF=4 (invalid - bad input data) -> should zero out
        dqf[8, :] = 4
        # Row 9: DQF=5 (invalid - algorithm failure) -> should zero out
        dqf[9, :] = 5

        cls.ds = _make_dataset(mask, dqf)
        cls.result = map_fdc_mask_to_confidence(cls.ds)
        cls.conf = cls.result["MaskConfidence"].values

    def test_good_quality_fire_preserves_confidence(self):
        """DQF=0 should preserve the mapped confidence value."""
        np.testing.assert_array_equal(self.conf[0, :], 1.0)
        np.testing.assert_array_equal(self.conf[1, :], 1.0)

    def test_good_quality_fire_free_land_preserves_confidence(self):
        """DQF=1 should preserve the mapped confidence value."""
        np.testing.assert_array_equal(self.conf[2, :], 1.0)
        np.testing.assert_array_equal(self.conf[3, :], 1.0)

    def test_opaque_cloud_zeros_confidence(self):
        """DQF=2 should zero out confidence regardless of mask code."""
        np.testing.assert_array_equal(self.conf[4, :], 0.0)
        np.testing.assert_array_equal(self.conf[5, :], 0.0)

    def test_surface_type_zeros_confidence(self):
        """DQF=3 should zero out confidence regardless of mask code."""
        np.testing.assert_array_equal(self.conf[6, :], 0.0)
        np.testing.assert_array_equal(self.conf[7, :], 0.0)

    def test_bad_input_data_zeros_confidence(self):
        """DQF=4 should zero out confidence regardless of mask code."""
        np.testing.assert_array_equal(self.conf[8, :], 0.0)

    def test_algorithm_failure_zeros_confidence(self):
        """DQF=5 should zero out confidence regardless of mask code."""
        np.testing.assert_array_equal(self.conf[9, :], 0.0)

    def test_bad_dqf_zeros_all_fire_categories(self):
        """
        Verify that even high-confidence mask codes (10, 11) are zeroed
        when DQF indicates bad quality.
        """
        # Build a dataset with various fire codes but all bad DQF
        mask = np.array([[10, 11, 12, 13, 14, 15, 30, 31, 32, 33]], dtype=np.int16)
        dqf = np.full((1, 10), 3, dtype=np.int16)  # all bad DQF
        ds = _make_dataset(mask, dqf)
        result = map_fdc_mask_to_confidence(ds)
        np.testing.assert_array_equal(result["MaskConfidence"].values, 0.0)

    def test_mixed_dqf_within_single_row(self):
        """
        Within a single row, good and bad DQF pixels coexist.
        Only good DQF pixels should retain their confidence.
        """
        mask = np.full((1, 10), 10, dtype=np.int16)
        dqf = np.array([[0, 1, 2, 3, 4, 5, 0, 1, 2, 0]], dtype=np.int16)
        ds = _make_dataset(mask, dqf)
        result = map_fdc_mask_to_confidence(ds)
        conf = result["MaskConfidence"].values[0]

        expected = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
                            dtype=np.float32)
        np.testing.assert_array_equal(conf, expected)
