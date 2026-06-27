from gofer.ortho import orthorectify
from gofer.remapper import map_fdc_mask_to_confidence 
from viz.gofer.ortho import (
    make_original_vs_terrain_corrected_gif,
    make_original_vs_terrain_corrected_gif2
)
import pickle
from pathlib import Path
import time
import xarray as xr

with open('temp/filelist.pkl', 'rb') as f:
    data = pickle.load(f)
    goes_filepaths = data['west'][:10]

    buffer = 0.1
    bbox = (
        data['lon_min'] - buffer,
        data['lat_min'] - buffer,
        data['lon_max'] + buffer,
        data['lat_max'] + buffer
    )

    dem_filepath = '/home/mgraca/Workspace/fire-spread/data/dem/SRTMGL3_NC.003_SRTMGL3_DEM_doy2000042000000_aid0001.tif'

print("Openining and combining datasets...", end=" ")
start_time = time.perf_counter()
goes_ds = xr.open_mfdataset(
    [p for p in goes_filepaths if Path(p).is_file()],
    concat_dim="time",
    combine="nested",
    drop_variables=["Area", "Temp", "Power"],
    decode_times=False,
    parallel=True,
    chunks={"time": 1, "y": 1500, "x": 2500}
)
print(f"complete. Time elapsed: {(time.perf_counter() - start_time):.1f}")
print(goes_ds)

print("Remapping mask to confidence values...", end=" ")
start_time = time.perf_counter()
remapped_ds = map_fdc_mask_to_confidence(goes_ds)
print(f"complete. Time elapsed: {(time.perf_counter() - start_time):.1f}")
print(remapped_ds)

print("Orthorectifying...", end=" ")
start_time = time.perf_counter()
ortho_ds = orthorectify(
    remapped_ds,
    dem_filepath=dem_filepath,
    bbox=bbox,
    data_vars=["MaskConfidence"],
)
print(f"complete. Time elapsed: {(time.perf_counter() - start_time):.1f}")
print(ortho_ds)

# viz -- sierra nevada
goes_f = '/home/mgraca/Workspace/fire-spread/tests/gofer/data/OR_ABI-L1b-RadC-M3C02_G16_s20171110002189_e20171110004562_c20171110004596.nc'
gif_path = make_original_vs_terrain_corrected_gif(
    goes_filepath=goes_f,
    ortho_ds=orthorectify(
        xr.open_dataset(goes_f),
        dem_filepath=dem_filepath,
        bbox=(-120.25, 37.25, -118.75, 38.50)
    ),
    variable="Rad",
    output_gif='out/ortho/sierra_goes_original_vs_terrain_corrected.gif',
    cmap="gray",
    duration_ms=900,
)

# viz -- fdc
goes_f = '/home/mgraca/Workspace/fire-spread/data/goes/noaa-goes17/ABI-L2-FDCC/2020/255/08/OR_ABI-L2-FDCC-M6_G17_s20202550801176_e20202550803549_c20202550804107.nc'
ds = map_fdc_mask_to_confidence(xr.open_dataset(goes_f))
gif_path = make_original_vs_terrain_corrected_gif2(
    original_ds=ds,
    terrain_corrected_ds=orthorectify(
        ds,
        dem_filepath=dem_filepath,
        bbox=bbox
    ),
    output_gif='out/ortho/fdc_goes_original_vs_terrain_corrected.gif',
    cmap="inferno",
)
