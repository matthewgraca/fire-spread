import xarray as xr
import pickle
import pandas as pd
from gofer.temporal_downsampler import aggregate
import numpy as np
from gofer.goes_utils import eval_and_save_nc


with open('temp/metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)
    dates = metadata['dates']

goes_save_dir = '/home/mgraca/Workspace/fire-spread/data/goes'
csv_path = 'temp/west_files.csv'
ds = aggregate(goes_save_dir, csv_path, 'temp/bobcat_2020/hourly', dates)
ds = eval_and_save_nc(ds, 'temp.nc', chunk_size=(1, 1500, 2500), chunks={'time': 1})
print(ds)

