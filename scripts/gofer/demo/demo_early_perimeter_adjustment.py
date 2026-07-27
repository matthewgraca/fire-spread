from gofer.early_perimeter_adjustment import *
import xarray as xr
import numpy as np
import pickle

with open('temp/metadata.pkl', 'rb') as f:
    data = pickle.load(f)
    dates = data['dates']

    buffer = 0.1
    bbox = (
        data['lon_min'] - buffer,
        data['lat_min'] - buffer,
        data['lon_max'] + buffer,
        data['lat_max'] + buffer
    )

    dem_filepath = (
        '/home/mgraca/Workspace/fire-spread/data/dem/'
        'SRTMGL3_NC.003_SRTMGL3_DEM_doy2000042000000_aid0001.tif'
    )

# start after aggregate()
ds = xr.open_dataset('temp/west_bobcat_2020_aggregated.nc', chunks={"time": 1})
#ds = xr.open_dataset('temp/east_bobcat_2020_aggregated.nc', chunks={"time": 1})
print(ds)

ortho_kwargs = {
    'dem_filepath' : dem_filepath,
    'bbox' : bbox
}
sf = get_scaling_factors(ds, ortho_kwargs=ortho_kwargs)
print(len(sf))
print(sf)

out = apply_scaling_factors(ds, sf)
print(out)

diff = out['MaskConfidence'] - ds['MaskConfidence']
max_diff = np.abs(diff).max()
print(max_diff.values)
