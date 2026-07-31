"""
Demo: composite GOES-West and GOES-East orthorectified datasets.

Assumes ortho'd datasets already exist (produced by earlier pipeline steps).
The composite function merges the two satellites by averaging their
MaskConfidence values at each timestep.
"""
import pickle
import xarray as xr
import pandas as pd
from gofer.composite import composite

# Load metadata to get the date range
with open('temp/metadata.pkl', 'rb') as f:
    data = pickle.load(f)
    dates = data['dates']

# Load orthorectified datasets (produced by prior pipeline steps)
west_ds = xr.open_dataset('temp/bobcat_2020/netcdf/west/ortho.nc', chunks='auto')
east_ds = xr.open_dataset('temp/bobcat_2020/netcdf/east/ortho.nc', chunks='auto')

# Composite: averages West and East MaskConfidence
ds = composite(west_ds, east_ds, dates, data_var='MaskConfidence')
print(ds)
print(f"Time steps: {ds.sizes['time']}")
print(f"Spatial shape: {ds.sizes['latitude']} x {ds.sizes['longitude']}")
