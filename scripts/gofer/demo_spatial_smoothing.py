import xarray as xr
from gofer.spatial_smoothing import smooth
import sys

ds = xr.open_dataset('temp/bobcat_2020_composited.nc')
print(ds)
smoothed_ds = smooth(ds, kernel_width_m=1_700)
print(smoothed_ds)
