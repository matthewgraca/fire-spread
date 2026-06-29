from gofer.ortho import orthorectify
from gofer.remapper import map_fdc_mask_to_confidence 
from gofer.composite import composite
from viz.gofer.ortho import (
    make_original_vs_terrain_corrected_gif,
    make_original_vs_terrain_corrected_gif2
)
import pickle
from pathlib import Path
import time
import xarray as xr
from pathlib import Path
import pandas as pd
import numpy as np

def open_and_combine_ds(goes_filepaths):
    print("Openining and combining datasets...", end=" ")
    start_time = time.perf_counter()
    ds = xr.open_mfdataset(
        [p for p in goes_filepaths if Path(p).is_file()],
        concat_dim="time",
        combine="nested",
        drop_variables=["Area", "Temp", "Power"],
        decode_times=False,
        parallel=True,
        chunks={"time": 1, "y": 1500, "x": 2500}
    )
    
    # decode_times=True sometimes plays out poorly, so we'll do it manually
    # ds.t.attrs['units'] = seconds since 2000-01-01 12:00:00
    origin = pd.Timestamp("2000-01-01 12:00:00")
    decoded_times = origin + pd.to_timedelta(ds["t"].values, unit="s")
    ds = ds.assign_coords(time=decoded_times)

    print(f"complete. Time elapsed: {(time.perf_counter() - start_time):.1f}")
    print(ds)
    return ds 

def remap(goes_ds):
    print("Remapping mask to confidence values...", end=" ")
    start_time = time.perf_counter()
    remapped_ds = map_fdc_mask_to_confidence(goes_ds)
    print(f"complete. Time elapsed: {(time.perf_counter() - start_time):.1f}")
    print(remapped_ds)
    return remapped_ds

def ortho(remapped_ds, dem_filepath, bbox):
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
    return ortho_ds

def comp(west_ortho_ds, east_ortho_ds, dates):
    print("Compositing...", end=" ")
    start_time = time.perf_counter()
    composite_ds = composite(
        west_ortho_ds,
        east_ortho_ds,
        dates,
        data_var='MaskConfidence'
    )
    print(f"complete. Time elapsed: {(time.perf_counter() - start_time):.1f}")
    print(composite_ds)
    return composite_ds

def save_nc(ds, save_path):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    '''
    ds.to_netcdf(
        str(save_path),
        mode="w",
        engine="netcdf4",
        encoding={
            "MaskConfidence": {
                "zlib": True,
                "complevel": 4,
                "dtype": "float32",
                "_FillValue": np.float32(np.nan),
            }
        }
    )
    '''
    ds.to_netcdf(
        str(save_path),
        mode="w",
        engine="netcdf4",
        encoding={
            name: {"_FillValue": np.nan}
            for name, da in ds.variables.items()
            if np.issubdtype(da.dtype, np.floating)
        }
    )

def main():
    with open('temp/filelist.pkl', 'rb') as f:
        data = pickle.load(f)
        ''' small sample to test with
        west_goes_filepaths = data['west'][:24]
        east_goes_filepaths = data['east'][:24]
        dates = data['dates'][:24]
        '''
        west_goes_filepaths = data['west']
        east_goes_filepaths = data['east']
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

    west_goes_ds = open_and_combine_ds(west_goes_filepaths)
    west_remapped_ds = remap(west_goes_ds)
    west_ortho_ds = ortho(west_remapped_ds, dem_filepath, bbox)
    save_nc(west_ortho_ds, 'temp/west_bobcat_2020.nc')

    east_goes_ds = open_and_combine_ds(east_goes_filepaths)
    east_remapped_ds = remap(east_goes_ds)
    east_ortho_ds = ortho(east_remapped_ds, dem_filepath, bbox)
    save_nc(west_ortho_ds, 'temp/east_bobcat_2020.nc')

    '''
    west_ortho_ds = xr.open_dataset('temp/west_bobcat_2020.nc')
    east_ortho_ds = xr.open_dataset('temp/east_bobcat_2020.nc')
    '''

    composite_ds = comp(west_ortho_ds, east_ortho_ds, dates)
    save_nc(composite_ds, save_path='out/bobcat_2020.nc')

    # viz -- sierra nevada orthorectification
    '''
    goes_f = (
        '/home/mgraca/Workspace/fire-spread/tests/gofer/data/'
        'OR_ABI-L1b-RadC-M3C02_G16_s20171110002189_e20171110004562_c20171110004596.nc'
    )
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
    '''

    # viz -- fdc remapping + orthorectification
    '''
    goes_f = (
        '/home/mgraca/Workspace/fire-spread/data/goes/'
        'noaa-goes17/ABI-L2-FDCC/2020/255/08/'
        'OR_ABI-L2-FDCC-M6_G17_s20202550801176_e20202550803549_c20202550804107.nc'
    )
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
    '''

if __name__=="__main__":
    main()
