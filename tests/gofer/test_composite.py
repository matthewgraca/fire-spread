import unittest
import pickle
import xarray as xr
import pandas as pd
from gofer.composite import composite

class TestComposite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open('tests/gofer/data/filelist.pkl', 'rb') as f:
            cls.pkl = pickle.load(f)
            cls.actual = composite(
                west_ds=xr.open_dataset('tests/gofer/data/west_bobcat_2020.nc'),
                east_ds=xr.open_dataset('tests/gofer/data/east_bobcat_2020.nc'),
                dates=cls.pkl['dates'],
                data_var='MaskConfidence'
            )

    def test_date_lengths_match(self):
        expected = self.pkl['dates']
        actual = pd.DatetimeIndex(self.actual['time'].values)
        self.assertTrue(len(expected) == len(actual))

    def test_dates_share_same_order_and_values(self):
        expected = self.pkl['dates']
        actual = pd.DatetimeIndex(self.actual['time'].values)
        self.assertTrue(expected.equals(actual))

    def test_dates_are_unique(self):
        expected = self.pkl['dates']
        actual = pd.DatetimeIndex(self.actual['time'].values)
        self.assertTrue(expected.is_unique)

    def test_ds_dates_are_unique(self):
        expected = self.pkl['dates']
        actual = pd.DatetimeIndex(self.actual['time'].values)
        self.assertTrue(actual.is_unique)

    def test_dates_are_monotonically_increasing(self):
        expected = self.pkl['dates']
        actual = pd.DatetimeIndex(self.actual['time'].values)
        self.assertTrue(expected.is_unique)

    def test_ds_dates_are_monotonically_increasing(self):
        expected = self.pkl['dates']
        actual = pd.DatetimeIndex(self.actual['time'].values)
        self.assertTrue(actual.is_unique)
