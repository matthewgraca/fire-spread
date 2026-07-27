import xarray as xr
from gofer.spatial_smoothing import smooth
import sys

ds = xr.open_dataset('temp/bobcat_2020/composited.nc')
print(ds)
smoothed_ds = smooth(ds, kernel_radius_m=1_700)
print(smoothed_ds)
