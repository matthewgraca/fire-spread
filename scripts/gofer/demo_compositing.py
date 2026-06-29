import pickle
import xarray as xr
from gofer.composite import composite

with open('temp/filelist.pkl', 'rb') as f:
    data = pickle.load(f)
    print(data.keys())

    west_file = data['west'][400]
    east_file = data['east'][400]
    dates = data['dates']

    buffer = 0.1
    bbox = (
        data['lon_min'] - buffer,
        data['lat_min'] - buffer,
        data['lon_max'] + buffer,
        data['lat_max'] + buffer
    )

ds = composite(
    xr.open_dataset(west_file, decode_times=False),
    xr.open_dataset(east_file, decode_times=False),
    dates[400],
    'Mask'
)

print(ds)
