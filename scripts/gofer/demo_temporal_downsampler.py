import xarray as xr
import pickle
import pandas as pd
from gofer.temporal_downsampler import aggregate


with open('temp/metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)
    dates = metadata['dates']
print(dates)

goes_save_dir = '/home/mgraca/Workspace/fire-spread/data/goes'
csv_path = 'temp/west_files.csv'
ds = aggregate(goes_save_dir, csv_path, dates)
print(ds)
print(ds['time'].values)
