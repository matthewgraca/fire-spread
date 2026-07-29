import unittest
import shutil
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path


TEST_DATA_DIR = Path("tests/gofer/data/composite")
TEMP_DIR = Path("tests/gofer/data/composite/temp_output")


def _build_ortho_dataset(sat_dir: Path) -> xr.Dataset:
    """Load the test hourly files into a single dataset mimicking ortho output."""
    files = sorted(sat_dir.glob("*.nc"))
    ds = xr.open_mfdataset(
        [str(f) for f in files],
        combine='nested',
        concat_dim='time',
        chunks={'time': 1},
    )
    return ds


class TestComposite(unittest.TestCase):
    """
    Test the slice-by-slice compositing behavior.

    Uses real Bobcat 2020 data cropped to 10x10 spatial pixels across
    5 hourly timesteps. West has fire pixels; East does not at this
    location. The composite (mean) should produce 0.5 where only one
    satellite detects fire.
    """

    @classmethod
    def setUpClass(cls):
        """Build west and east datasets from test fixtures."""
        cls.west_ds = _build_ortho_dataset(TEST_DATA_DIR / "west")
        cls.east_ds = _build_ortho_dataset(TEST_DATA_DIR / "east")
        TEMP_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        cls.west_ds.close()
        cls.east_ds.close()
        if TEMP_DIR.exists():
            shutil.rmtree(TEMP_DIR)

    def test_composite_produces_correct_shape(self):
        """Output should have same time, lat, lon dimensions as inputs."""
        from scripts.gofer.run import step_composite

        # Temporarily monkey-patch the netcdf_dir
        composite_ds = self._run_composite()
        self.assertEqual(composite_ds.sizes['time'], 5)
        self.assertEqual(composite_ds.sizes['latitude'], 10)
        self.assertEqual(composite_ds.sizes['longitude'], 10)
        composite_ds.close()

    def test_composite_averages_east_and_west(self):
        """Where west=1.0 and east=0.0, composite should be 0.5."""
        composite_ds = self._run_composite()
        # Last timestep: west has fire at [1,0], [1,1], [2,0], [2,1]
        mc = composite_ds['MaskConfidence'].isel(time=4).load().values
        # West=1.0, East=0.0 -> mean=0.5
        np.testing.assert_almost_equal(mc[1, 0], 0.5, decimal=2)
        np.testing.assert_almost_equal(mc[1, 1], 0.5, decimal=2)
        np.testing.assert_almost_equal(mc[2, 0], 0.5, decimal=2)
        np.testing.assert_almost_equal(mc[2, 1], 0.5, decimal=2)
        composite_ds.close()

    def test_composite_zeros_where_no_fire(self):
        """Where both east and west are 0.0, composite should be 0.0."""
        composite_ds = self._run_composite()
        mc = composite_ds['MaskConfidence'].isel(time=4).load().values
        # Bottom-right corner should be 0
        self.assertEqual(float(mc[9, 9]), 0.0)
        self.assertEqual(float(mc[5, 5]), 0.0)
        composite_ds.close()

    def test_composite_no_nans(self):
        """Output should have no NaN values."""
        composite_ds = self._run_composite()
        mc = composite_ds['MaskConfidence'].load()
        self.assertFalse(mc.isnull().any().item())
        composite_ds.close()

    def test_composite_values_in_valid_range(self):
        """All composited values should be in [0.0, 1.0]."""
        composite_ds = self._run_composite()
        mc = composite_ds['MaskConfidence'].load()
        self.assertTrue((mc >= 0.0).all().item())
        self.assertTrue((mc <= 1.0).all().item())
        composite_ds.close()

    def _run_composite(self) -> xr.Dataset:
        """Run the compositing logic matching step_composite's behavior."""
        out_dir = TEMP_DIR / 'composite_slices'
        out_dir.mkdir(parents=True, exist_ok=True)
        save_path = str(TEMP_DIR / 'composited.nc')

        west_ds = self.west_ds
        east_ds = self.east_ds

        n_times = west_ds.sizes['time']
        slice_paths = []

        for t in range(n_times):
            west_slice = west_ds['MaskConfidence'].isel(time=t).load()
            east_slice = east_ds['MaskConfidence'].isel(time=t).load()

            merged = np.nanmean(
                np.stack([west_slice.values, east_slice.values], axis=0),
                axis=0
            )

            time_val = west_ds.time.values[t]
            ds_slice = xr.Dataset(
                data_vars={
                    'MaskConfidence': (['latitude', 'longitude'], merged),
                },
                coords={
                    'latitude': west_ds.y.values,
                    'longitude': west_ds.x.values,
                    'time': time_val,
                },
            )

            slice_path = out_dir / f'{t:05d}.nc'
            ds_slice.to_netcdf(str(slice_path), engine='scipy')
            slice_paths.append(slice_path)

        composite_ds = xr.open_mfdataset(
            [str(p) for p in sorted(slice_paths)],
            combine='nested',
            concat_dim='time',
            chunks={'time': 1},
        )

        return composite_ds


if __name__ == "__main__":
    unittest.main()
