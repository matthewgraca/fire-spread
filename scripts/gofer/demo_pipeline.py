from gofer.ingest import download, read_calfire_geojson
from gofer.ortho import orthorectify
from gofer.remapper import map_fdc_mask_to_confidence 
from gofer.composite import composite
from viz.gofer.ortho import (
    make_original_vs_terrain_corrected_gif,
    make_original_vs_terrain_corrected_gif2
)
from gofer.spatial_smoothing import smooth
from gofer.temporal_downsampler import aggregate
from gofer.early_perimeter_adjustment import *
from gofer.goes_utils import eval_and_save_nc
import pickle
from pathlib import Path
import time
import xarray as xr
import pandas as pd
import numpy as np
import sys
from argparse import ArgumentParser
import rioxarray

parser = ArgumentParser(description='GOFER pipeline')
group = parser.add_mutually_exclusive_group()
group.add_argument('--ingest', action='store_true', help='Start from ingest')
group.add_argument('--aggregate', action='store_true', help='Start from aggregate')
group.add_argument('--scale', action='store_true', help='Start from scale')
group.add_argument('--ortho', action='store_true', help='Start from ortho')
group.add_argument('--composite', action='store_true', help='Start from composite')
group.add_argument('--smooth', action='store_true', help='Start from smooth')
parser.add_argument('-f', '--fire', type=str, default='bobcat', help='Name of the fire to ingest')
parser.add_argument('-y', '--year', type=int, default=2020, help='Year of the fire (to disambiguate fires that may share the same name)')
args = parser.parse_args()

pipeline_opts = {
    'ingest' : args.ingest,
    'aggregate' : args.aggregate,
    'scale' : args.scale,
    'ortho' : args.ortho,
    'composite' : args.composite,
    'smooth' : args.smooth 
}
if True not in set(list(pipeline_opts.values())):
    raise ValueError('Define at least one part of the pipeline to start on.')
pipeline = [
    'ingest',
    'aggregate',
    'scale',
    'ortho',
    'composite',
    'smooth',
]
run_pipeline = [True] * len(pipeline) 
for i, action in enumerate(pipeline):
    if pipeline_opts[action] is False:
        run_pipeline[i] = False
    else:
        break
print('Pipeline active:')
print([f'{x}: {y}' for x, y in zip(pipeline, run_pipeline)])

def aggregation(goes_save_dir, csv_path, temp_dir, dates, fire_name):
    print("Opening, remapping, combining, and temporally aligning datasets...", end=" ")
    start_time = time.perf_counter()
    ds = aggregate(goes_save_dir, csv_path, temp_dir, dates, fire_name=fire_name)
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
        data_var="MaskConfidence",
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

def smoothing(ds):
    print(f"Convolving over original raster...", end=" ")
    start_time = time.perf_counter()

    smoothed_ds = smooth(ds, kernel_width_m=1700)

    print(f"complete. Time elapsed: {(time.perf_counter() - start_time):.1f}")
    return smoothed_ds



def open_ds(path, data_var='MaskConfidence', chunks={"time": 1}):
    # opens the dataset in chunks and casts the data_var to float16
    # NOTE now obsolete
    ds = xr.open_dataset(path, chunks=chunks)
    return ds


def main():
    if run_pipeline[0]:
        calfire_geojson_path = "data/calfire/California_Historic_Fire_Perimeters_-4891938132824355098.geojson"
        gdf = read_calfire_geojson(calfire_geojson_path)
        fire = gdf.loc[gdf['FIRE_NAME'] == args.fire.upper()]

        fire_attrs = {
            'start' : fire['ALARM_DATE'].item(),
            'end' : fire['CONT_DATE'].item(),
            'fire_name' : str(fire['FIRE_NAME'].item()),
            'fire_year' : int(fire['YEAR_'].item()),
            'fire_acres' : int(fire['GIS_ACRES'].item())
        }
        fire_bbox = {
            'lon_min' : float(fire['bbox_min_lon'].item()),
            'lon_max' : float(fire['bbox_max_lon'].item()),
            'lat_min' : float(fire['bbox_min_lat'].item()),
            'lat_max' : float(fire['bbox_max_lat'].item())
        }

        download(
            **fire_attrs,
            goes_save_dir='/home/mgraca/Workspace/fire-spread/data/goes',
            metadata_save_dir=f'temp/{args.fire}_{args.year}',
            subhourly=True,
            **fire_bbox
        )
    else:
        pass

    with open(f'temp/{args.fire}_{args.year}/metadata.pkl', 'rb') as f:
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


    # open, remap, ortho each satellite
    # aggregate
    if run_pipeline[1]:
        west_goes_ds = aggregation(
            goes_save_dir='/home/mgraca/Workspace/fire-spread/data/goes',
            csv_path=f'/home/mgraca/Workspace/fire-spread/temp/{args.fire}_{args.year}/west_files.csv',
            temp_dir=f'temp/{args.fire}_{args.year}/west/hourly',
            dates=dates,
            fire_name=args.fire.upper()
        )
        west_goes_ds = eval_and_save_nc(
            west_goes_ds, 
            chunk_size=(1, 1500, 2500),
            save_path=f'temp/{args.fire}_{args.year}/west/aggregated.nc',
            chunks='auto',
            desc='west aggregation'
        )

        east_goes_ds = aggregation(
            goes_save_dir='/home/mgraca/Workspace/fire-spread/data/goes',
            csv_path=f'/home/mgraca/Workspace/fire-spread/temp/{args.fire}_{args.year}/east_files.csv',
            temp_dir=f'temp/{args.fire}_{args.year}/east/hourly',
            dates=dates,
            fire_name=args.fire.upper()
        )
        east_goes_ds = eval_and_save_nc(
            east_goes_ds,
            chunk_size=(1, 1500, 2500),
            save_path=f'temp/{args.fire}_{args.year}/east/aggregated.nc',
            chunks='auto',
            desc='east aggregation'
        )
    else:
        west_goes_ds = open_ds(f'temp/{args.fire}_{args.year}/west/aggregated.nc', chunks='auto')
        east_goes_ds = open_ds(f'temp/{args.fire}_{args.year}/east/aggregated.nc', chunks='auto')

    # cummax and select timestep to generate perimeter for
    # may get folded into aggregate, depending on if we actually need the original
    # a compromise could be another variable, like
    # MaskConfidenceBurnedArea vs MaskConfidenceActiveFire.
    print(west_goes_ds)
    print(east_goes_ds)

    # scale
    if run_pipeline[2]:
        west_sf = get_scaling_factors(
            west_goes_ds,
            ortho_kwargs={'dem_filepath' : dem_filepath, 'bbox': bbox}
        )
        west_scaled_ds = apply_scaling_factors(west_goes_ds, west_sf)
        west_scaled_ds = eval_and_save_nc(
            west_scaled_ds,
            chunk_size=(1, 1500, 2500),
            save_path=f'temp/{args.fire}_{args.year}/west/scaled.nc',
            chunks={'time': 1},
            desc='west scale factors'
        )

        east_sf = get_scaling_factors(
            east_goes_ds,
            ortho_kwargs={'dem_filepath' : dem_filepath, 'bbox': bbox}
        )
        east_scaled_ds = apply_scaling_factors(east_goes_ds, east_sf)
        east_scaled_ds = eval_and_save_nc(
            east_scaled_ds,
            chunk_size=(1, 1500, 2500),
            save_path=f'temp/{args.fire}_{args.year}/east/scaled.nc',
            chunks={'time': 1},
            desc='east scale factors'
        )
    else:
        west_scaled_ds = open_ds(f'temp/{args.fire}_{args.year}/west/scaled.nc')
        east_scaled_ds = open_ds(f'temp/{args.fire}_{args.year}/east/scaled.nc')

    # NOTE we currently only care about the final fire perimeter from here on out
    west_scaled_ds = west_scaled_ds.isel(time=[-1])
    east_scaled_ds = east_scaled_ds.isel(time=[-1])

    # ortho
    if run_pipeline[3]:
        west_ortho_ds = ortho(west_scaled_ds, dem_filepath, bbox)
        west_ortho_ds = eval_and_save_nc(
            west_ortho_ds,
            save_path=f'temp/{args.fire}_{args.year}/west/ortho.nc',
            chunks='auto',
            desc='west orthorectification'
        )
        east_ortho_ds = ortho(east_scaled_ds, dem_filepath, bbox)
        east_ortho_ds = eval_and_save_nc(
            east_ortho_ds,
            save_path=f'temp/{args.fire}_{args.year}/east/ortho.nc',
            chunks='auto',
            desc='east orthorectification'
        )
    else:
        west_ortho_ds = open_ds(f'temp/{args.fire}_{args.year}/west/ortho.nc', chunks='auto')
        east_ortho_ds = open_ds(f'temp/{args.fire}_{args.year}/east/ortho.nc', chunks='auto')


    # composite the two into one
    if run_pipeline[4]:
        composite_ds = comp(west_ortho_ds, east_ortho_ds, dates)
        composite_ds = eval_and_save_nc(
            composite_ds, 
            save_path=f'temp/{args.fire}_{args.year}/composited.nc',
            chunks='auto',
            desc='west compositing'
        )
    else:
        composite_ds = open_ds(f'temp/{args.fire}_{args.year}/composited.nc', chunks='auto')


    # apply smooth edges 
    if run_pipeline[5]:
        smoothed_ds = smoothing(composite_ds)
        smoothed_ds = eval_and_save_nc(
            smoothed_ds,
            save_path=f'out/{args.fire}_{args.year}_gofer.nc',
            chunks='auto',
            desc='east compositing'
        )
    else:
        smoothed_ds = open_ds(f'out/{args.fire}_{args.year}_gofer.nc', chunks='auto')


    ''' crack at 50m resampling
    ds_50m = (smoothed_ds
        .rio.set_spatial_dims(x_dim='latitude', y_dim='longitude')
        .rio.write_crs('EPSG:4326')
        .rio.reproject('EPSG:3310', resolution=50, resampling='nearest')
        .rio.reproject('EPSG:4326')
    )
    ds_50m = eval_and_save_nc(
        ds_50m,
        save_path=f'temp/{args.fire}_2020_downscaled.nc',
        chunks='auto',
        desc='50m downscaling'
    )
    '''

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
