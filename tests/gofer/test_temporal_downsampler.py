import unittest
import pickle
import xarray as xr
import pandas as pd
from gofer.temporal_downsampler import aggregate

class TestTemporalDownsampler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open('tests/gofer/data/metadata.pkl', 'rb') as f:
            cls.pkl = pickle.load(f)
            cls.dates = pd.DatetimeIndex(cls.pkl['dates'], tz='UTC')
            cls.actual = aggregate(
                goes_save_dir='/home/mgraca/Workspace/fire-spread/data/goes',
                csv_path='tests/gofer/data/goes_files.csv',
                dates=cls.pkl['dates']
            )
    def test_date_lengths_match(self):
        expected = self.dates
        actual = pd.DatetimeIndex(self.actual['time'].values, tz='UTC')
        self.assertTrue(len(expected) == len(actual))

    def test_dates_share_same_order_and_values(self):
        expected = self.dates
        actual = pd.DatetimeIndex(self.actual['time'].values, tz='UTC')
        self.assertTrue(
            expected.equals(actual),
            f'Actual: {actual}, Expected: {expected}'
        )

    def test_dates_are_unique(self):
        expected = self.dates
        actual = pd.DatetimeIndex(self.actual['time'].values, tz='UTC')
        self.assertTrue(expected.is_unique)

    def test_ds_dates_are_unique(self):
        expected = self.dates
        actual = pd.DatetimeIndex(self.actual['time'].values, tz='UTC')
        self.assertTrue(actual.is_unique)

    def test_dates_are_monotonically_increasing(self):
        expected = self.dates
        actual = pd.DatetimeIndex(self.actual['time'].values, tz='UTC')
        self.assertTrue(expected.is_unique)

    def test_ds_dates_are_monotonically_increasing(self):
        expected = self.dates
        actual = pd.DatetimeIndex(self.actual['time'].values, tz='UTC')
        self.assertTrue(actual.is_unique)
