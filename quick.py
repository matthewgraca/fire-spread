import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path
from gofer.goes_utils import *
from viz.gofer.goes_plotting_utils import *
import xarray as xr
import pickle

ds = xr.open_dataset('out/bobcat_2020_gofer.nc')
print(ds['MaskConfidence'].min().values)
print(ds['MaskConfidence'].max().values)
print(np.unique(ds['MaskConfidence'].round(decimals=2)))
