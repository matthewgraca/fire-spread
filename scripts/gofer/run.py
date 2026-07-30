"""
GOFER Pipeline — Unified Runner

Runs the full GOFER pipeline (ingest + processing) for all fires in a
manifest, configured via a YAML file.

Usage:
    python scripts/gofer/run.py --config configs/gofer.yaml
    python scripts/gofer/run.py --config configs/gofer.yaml --skip-ingest
    python scripts/gofer/run.py --config configs/gofer.yaml --only-ingest
"""
import sys
import gc
import pickle
import shutil
import time
from argparse import ArgumentParser
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
import yaml
from tqdm import tqdm

from gofer.composite import composite
from gofer.early_perimeter_adjustment import apply_scaling_factors, get_scaling_factors
from gofer.goes_utils import eval_and_save_nc
from gofer.ingest import download, read_calfire_geojson
from gofer.ortho import orthorectify
from gofer.postprocess import binarize, round_to, trim_inactive_timesteps
from gofer.spatial_smoothing import smooth
from gofer.temporal_downsampler import aggregate
from gofer.vectorize import raster_to_polygon
from gofer.style import Style

BBOX_BUFFER = 0.1
S = Style()


def load_manifest(manifest_path: str) -> pd.DataFrame:
    """
    Load and validate the fire manifest CSV.

    Expected columns: state, year, fire_name
    """
    df = pd.read_csv(manifest_path)
    required_cols = {'state', 'year', 'fire_name'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")
    df['fire_name'] = df['fire_name'].str.upper()
    return df


def lookup_fire(gdf: gpd.GeoDataFrame, fire_name: str, year: int):
    """
    Look up a fire from the CalFire GeoDataFrame by name and year.

    Returns the first matching row as a Series.
    """
    match = gdf.loc[
        (gdf['FIRE_NAME'] == fire_name) & (gdf['YEAR_'] == year)
    ]
    if len(match) == 0:
        raise ValueError(f"Fire '{fire_name}' ({year}) not found in CalFire data.")
    if len(match) > 1:
        tqdm.write(f"  Warning: multiple matches for '{fire_name}' ({year}), using first.")
    return match.iloc[0]


# --- Config loading ---

def load_config(config_path: str) -> dict:
    """Load and validate the YAML config file."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    required_keys = ['manifest', 'calfire_geojson', 'goes_dir', 'temp_dir', 'out_dir', 'dem']
    missing = [k for k in required_keys if k not in cfg]
    if missing:
        raise ValueError(f"Config missing required keys: {missing}")

    # Defaults
    cfg.setdefault('clean', False)
    cfg.setdefault('workers', 12)
    return cfg


# --- Argument parsing ---

def parse_args():
    parser = ArgumentParser(
        description='GOFER Pipeline — Ingest and process GOES fire data.'
    )
    parser.add_argument(
        '--config', type=str, required=True,
        help='Path to YAML config file.'
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--skip-ingest', action='store_true',
        help='Skip the ingest phase (assumes data is already downloaded).'
    )
    group.add_argument(
        '--only-ingest', action='store_true',
        help='Only run the ingest phase (download data, then stop).'
    )
    return parser.parse_args()


# --- Ingest phase ---

def run_ingest(manifest: pd.DataFrame, calfire_gdf: gpd.GeoDataFrame, cfg: dict):
    """Download GOES data for all fires in the manifest."""
    tqdm.write(S.phase("Phase 1: Ingesting GOES data"))

    total = len(manifest)
    start_time = time.time()

    for i, (_, fire_row) in enumerate(manifest.iterrows(), start=1):
        fire_name = fire_row['fire_name']
        fire_year = int(fire_row['year'])
        fire_id = f"{fire_name.lower().replace(' ', '_')}_{fire_year}"
        temp_dir = str(Path(cfg['temp_dir']) / fire_id)

        tqdm.write(f"\n{S.fire_header(i, total, fire_name, fire_year)}")

        try:
            fire = lookup_fire(calfire_gdf, fire_name, fire_year)

            download(
                start=fire['ALARM_DATE'],
                end=fire['CONT_DATE'],
                fire_name=str(fire['FIRE_NAME']),
                fire_year=int(fire['YEAR_']),
                fire_acres=int(fire['GIS_ACRES']),
                goes_save_dir=cfg['goes_dir'],
                metadata_save_dir=temp_dir,
                subhourly=True,
                lon_min=float(fire['bbox_min_lon']),
                lon_max=float(fire['bbox_max_lon']),
                lat_min=float(fire['bbox_min_lat']),
                lat_max=float(fire['bbox_max_lat']),
            )
            tqdm.write(S.success(fire_name, fire_year))
        except Exception as e:
            tqdm.write(S.error(fire_name, fire_year, str(e)))
            import traceback
            tqdm.write(S.traceback(traceback.format_exc()))

        elapsed = time.time() - start_time
        avg_per_fire = elapsed / i
        remaining = avg_per_fire * (total - i)
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
        remaining_str = time.strftime("%H:%M:%S", time.gmtime(remaining))
        tqdm.write(S.timing(i, total, elapsed_str, remaining_str))


# --- Processing phase ---

def run_processing(manifest: pd.DataFrame, calfire_gdf: gpd.GeoDataFrame, cfg: dict):
    """Run the full processing pipeline for all fires in the manifest."""
    tqdm.write("")
    tqdm.write(S.phase("Phase 2: Processing pipeline"))

    total = len(manifest)
    start_time = time.time()

    for i, (_, fire_row) in enumerate(manifest.iterrows(), start=1):
        fire_name = fire_row['fire_name']
        fire_year = int(fire_row['year'])
        fire_id = f"{fire_name.lower()}_{fire_year}"

        tqdm.write(f"\n{S.fire_header(i, total, fire_name, fire_year)}")

        try:
            process_fire(fire_name, fire_year, fire_id, calfire_gdf, cfg)
            tqdm.write(S.success(fire_name, fire_year))
        except Exception as e:
            tqdm.write(S.error(fire_name, fire_year, str(e)))
            import traceback
            tqdm.write(S.traceback(traceback.format_exc()))

        elapsed = time.time() - start_time
        avg_per_fire = elapsed / i
        remaining = avg_per_fire * (total - i)
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
        remaining_str = time.strftime("%H:%M:%S", time.gmtime(remaining))
        tqdm.write(S.timing(i, total, elapsed_str, remaining_str))


def process_fire(
    fire_name: str,
    fire_year: int,
    fire_id: str,
    calfire_gdf: gpd.GeoDataFrame,
    cfg: dict,
):
    """Run all processing steps for a single fire."""
    fire = lookup_fire(calfire_gdf, fire_name, fire_year)

    temp_dir = str(Path(cfg['temp_dir']) / fire_id)
    netcdf_dir = f'{temp_dir}/netcdf'
    out_dir = cfg['out_dir']
    dem = cfg['dem']

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    fire_meta = {
        'fire_name': fire_name,
        'fire_year': fire_year,
        'fire_acres': int(fire['GIS_ACRES']),
    }

    # Load ingest metadata
    metadata_path = Path(temp_dir) / 'metadata.pkl'
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata not found at {metadata_path}. "
            f"Run ingest first for {fire_name} ({fire_year})."
        )

    with open(metadata_path, 'rb') as f:
        data = pickle.load(f)
        dates = data['dates']
        bbox = (
            data['lon_min'] - BBOX_BUFFER,
            data['lat_min'] - BBOX_BUFFER,
            data['lon_max'] + BBOX_BUFFER,
            data['lat_max'] + BBOX_BUFFER,
        )

    ### FIXME TEMPORARY SKIP OF AGGREGATE, REMOVE
    '''
    # [1/6] Aggregate
    tqdm.write(S.step(1, 6, "Aggregating..."))
    west_ds, east_ds = step_aggregate(cfg['goes_dir'], temp_dir, netcdf_dir, dates, fire_name, cfg['workers'])

    # [1/6] Aggregate (TEMP: skip, load from disk)
    tqdm.write(S.step(1, 6, "Aggregating... (skipped, loading from disk)"))
    west_ds = xr.open_dataset(f'{netcdf_dir}/west/aggregated.nc', chunks={'time': 1})
    east_ds = xr.open_dataset(f'{netcdf_dir}/east/aggregated.nc', chunks={'time': 1})

    ####

    # [2/6] Scale
    tqdm.write(S.step(2, 6, "Scaling early perimeters..."))
    west_ds, east_ds = step_scale(west_ds, east_ds, dem, bbox, netcdf_dir)

    # [3/6] Ortho
    tqdm.write(S.step(3, 6, "Orthorectifying..."))
    west_ds, east_ds = step_ortho(west_ds, east_ds, dem, bbox, netcdf_dir, cfg['workers'])
    '''
    ### FIXME remove when done
    west_ds = xr.open_dataset(f'{netcdf_dir}/west/ortho.nc', chunks={'time': 1})
    east_ds = xr.open_dataset(f'{netcdf_dir}/east/ortho.nc', chunks={'time': 1})
    ###

    # [4/6] Composite
    tqdm.write(S.step(4, 6, "Compositing..."))
    sys.stdout.flush()
    ds = step_composite(west_ds, east_ds, dates, netcdf_dir)

    # [5/6] Smooth
    tqdm.write(S.step(5, 6, "Smoothing..."))
    sys.stdout.flush()
    ds = step_smooth(ds, netcdf_dir)

    # [6/6] Final
    tqdm.write(S.step(6, 6, "Final processing..."))
    sys.stdout.flush()
    calfire_ref = calfire_gdf.loc[
        (calfire_gdf['FIRE_NAME'] == fire_name) &
        (calfire_gdf['YEAR_'] == fire_year)
    ].to_crs(epsg=4326)
    calfire_ref = calfire_ref if len(calfire_ref) > 0 else None

    step_final(ds, fire_meta, out_dir, calfire_gdf=calfire_ref)

    # Cleanup intermediates
    if cfg['clean'] and Path(netcdf_dir).exists():
        tqdm.write(S.substep("Cleaning intermediate files...", last_step=True))
        shutil.rmtree(netcdf_dir)


# --- Pipeline steps ---

def step_aggregate(goes_save_dir: str, temp_dir: str, netcdf_dir: str,
                   dates: pd.DatetimeIndex, fire_name: str, threads: int):
    """Remap, temporally downsample, and aggregate both satellites."""
    results = {}
    for sat in ['west', 'east']:
        tqdm.write(S.substep(f"Aggregating GOES-{sat.capitalize()}..."))
        save_path = str(Path(netcdf_dir) / sat / 'aggregated.nc')
        ds = aggregate(
            goes_save_dir=goes_save_dir,
            csv_path=str(Path(temp_dir) / f'{sat}_files.csv'),
            temp_dir=str(Path(netcdf_dir) / sat / 'hourly'),
            dates=dates,
            fire_name=fire_name,
            verbose=True,
            max_workers=threads,
        )
        ds = eval_and_save_nc(
            ds,
            chunk_size=(1, 1500, 2500),
            save_path=save_path,
            chunks='auto',
            desc=f'{sat} aggregation',
            verbose=True,
        )
        tqdm.write(S.substep(f"Saved to {save_path}"))
        results[sat] = ds
    return results['west'], results['east']


def step_scale(west_ds, east_ds, dem_filepath: str, bbox: tuple, netcdf_dir: str):
    """Compute and apply early perimeter scaling factors."""
    results = {}
    for sat, ds in [('west', west_ds), ('east', east_ds)]:
        tqdm.write(S.substep(f"Scaling GOES-{sat.capitalize()}..."))
        save_path = str(Path(netcdf_dir) / sat / 'scaled.nc')
        sf = get_scaling_factors(
            ds,
            ortho_kwargs={'dem_filepath': dem_filepath, 'bbox': bbox},
            show_progress=True,
        )
        scaled_ds = apply_scaling_factors(ds, sf)
        ds.close()
        scaled_ds = eval_and_save_nc(
            scaled_ds,
            chunk_size=(1, 1500, 2500),
            save_path=save_path,
            chunks={'time': 1},
            desc=f'{sat} scaling',
            verbose=True,
        )
        tqdm.write(S.substep(f"Saved to {save_path}"))
        results[sat] = scaled_ds
        gc.collect()
    return results['west'], results['east']


def step_ortho(west_ds, east_ds, dem_filepath: str, bbox: tuple, netcdf_dir: str, max_workers: int = 12):
    """Orthorectify both satellite datasets using parallel workers."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from gofer.ortho import make_ortho_map

    results = {}
    for sat, ds in [('west', west_ds), ('east', east_ds)]:
        tqdm.write(S.substep(f"Orthorectifying GOES-{sat.capitalize()}..."))
        save_path = str(Path(netcdf_dir) / sat / 'ortho.nc')

        # Build the ortho map once
        ortho_map = make_ortho_map(
            goes_ds=ds,
            dem_filepath=dem_filepath,
            bbox=bbox,
        )

        # Save ortho map for worker processes
        slice_dir = Path(netcdf_dir) / sat / 'ortho_slices'
        slice_dir.mkdir(parents=True, exist_ok=True)
        ortho_map_path = str(slice_dir / 'ortho_map.nc')
        encoding = {v: {'dtype': 'float32'} for v in ortho_map.coords
                    if ortho_map.coords[v].dtype.kind == 'f'}
        ortho_map.to_netcdf(ortho_map_path, engine='scipy', encoding=encoding)

        source_path = str(Path(netcdf_dir) / sat / 'scaled.nc')
        n_times = ds.sizes['time']
        ds.close()

        # Parallel orthorectification
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    _ortho_slice, source_path, ortho_map_path, t, str(slice_dir)
                ): t
                for t in range(n_times)
            }
            slice_paths = [None] * n_times
            with tqdm(total=n_times, desc=f"Orthorectifying {sat}",
                      leave=False, delay=1) as pbar:
                for future in as_completed(futures):
                    t = futures[future]
                    slice_paths[t] = future.result()
                    pbar.update(1)

        # Combine into final ortho file
        ortho_ds = xr.open_mfdataset(
            slice_paths,
            combine='nested',
            data_vars='all',
            concat_dim='time',
            chunks={'time': 1},
        )
        ortho_ds = eval_and_save_nc(
            ortho_ds,
            save_path=save_path,
            chunks={'time': 1},
            desc=f'{sat} orthorectification',
            verbose=True,
        )

        shutil.rmtree(slice_dir)
        gc.collect()

        tqdm.write(S.substep(f"Saved to {save_path}"))
        results[sat] = ortho_ds
    return results['west'], results['east']


def _ortho_slice(source_path: str, ortho_map_path: str, t: int, out_dir: str) -> str:
    """Orthorectify a single time slice. Runs in a worker process."""
    import xarray as xr
    import numpy as np
    from gofer.ortho import apply_ortho_map
    from gofer.goes_utils import MC_ENCODING

    ortho_map = xr.open_dataset(ortho_map_path)
    ds = xr.open_dataset(source_path)
    ds_t = ds.isel(time=t).load()
    ds.close()

    ortho_t = apply_ortho_map(
        ds_t.expand_dims('time'),
        ortho_map,
        data_var="MaskConfidence",
    )

    # Only keep MaskConfidence — drop static geometry variables
    ortho_t = ortho_t[['MaskConfidence']]

    slice_path = f"{out_dir}/{t:05d}.nc"
    encoding = {name: {'dtype': 'float32'} for name in ortho_t.coords
                if ortho_t.coords[name].dtype.kind == 'f'}
    encoding['MaskConfidence'] = MC_ENCODING
    ortho_t.to_netcdf(slice_path, engine='scipy', encoding=encoding)
    ortho_t.close()
    ortho_map.close()

    return slice_path


def step_composite(west_ds, east_ds, dates: pd.DatetimeIndex, netcdf_dir: str):
    """Composite East and West into a single dataset, one time slice at a time."""
    import h5netcdf

    tqdm.write(S.substep("Compositing East and West..."))
    save_path = str(Path(netcdf_dir) / 'composited.nc')

    n_times = west_ds.sizes['time']
    lat_vals = west_ds.latitude.values
    lon_vals = west_ds.longitude.values
    time_vals = west_ds.time.values

    keep_attrs = {'fire_name', 'description', 'active_fire', 'fire_perimeter'}
    carried_attrs = {k: west_ds.attrs[k] for k in west_ds.attrs if k in keep_attrs}

    with h5netcdf.File(save_path, 'w') as f:
        f.dimensions = {'time': n_times, 'latitude': len(lat_vals), 'longitude': len(lon_vals)}
        f.create_variable('time', ('time',), data=time_vals.astype('int64'))
        f.create_variable('latitude', ('latitude',), data=lat_vals.astype('float32'))
        f.create_variable('longitude', ('longitude',), data=lon_vals.astype('float32'))
        mc_var = f.create_variable(
            'MaskConfidence', ('time', 'latitude', 'longitude'),
            dtype='int8', fillvalue=np.int8(-1),
        )
        mc_var.attrs['scale_factor'] = np.float32(0.01)
        mc_var.attrs['add_offset'] = np.float32(0.0)


        for t in range(n_times):
            west_slice = west_ds['MaskConfidence'].isel(time=t).load().values
            east_slice = east_ds['MaskConfidence'].isel(time=t).load().values
            merged = np.nanmean(
                np.stack([west_slice, east_slice], axis=0), axis=0
            )
            mc_var[t, :, :] = np.clip(merged / 0.01, 0, 100).astype(np.int8)
            del west_slice, east_slice, merged

        f.attrs['pipeline'] = 'composited'
        for k, v in carried_attrs.items():
            f.attrs[k] = v

    west_ds.close()
    east_ds.close()
    gc.collect()

    composite_ds = xr.open_dataset(save_path, chunks={'time': 1})
    tqdm.write(S.substep(f"Saved to {save_path}"))
    return composite_ds


def step_smooth(ds, netcdf_dir: str):
    """Apply spatial smoothing, one time slice at a time."""
    import h5netcdf
    from gofer.spatial_smoothing import smooth

    tqdm.write(S.substep("Smoothing..."))
    save_path = str(Path(netcdf_dir) / 'smoothed.nc')

    n_times = ds.sizes['time']
    lat_vals = ds.latitude.values
    lon_vals = ds.longitude.values
    time_vals = ds.time.values

    # Write directly to final file, one slice at a time
    with h5netcdf.File(save_path, 'w') as f:
        f.dimensions = {'time': n_times, 'latitude': len(lat_vals), 'longitude': len(lon_vals)}
        f.create_variable('time', ('time',), data=time_vals.astype('int64'))
        f.create_variable('latitude', ('latitude',), data=lat_vals.astype('float32'))
        f.create_variable('longitude', ('longitude',), data=lon_vals.astype('float32'))
        mc_var = f.create_variable(
            'MaskConfidence', ('time', 'latitude', 'longitude'),
            dtype='int8', fillvalue=np.int8(-1),
        )
        mc_var.attrs['scale_factor'] = np.float32(0.01)
        mc_var.attrs['add_offset'] = np.float32(0.0)


        for t in range(n_times):
            ds_t = ds.isel(time=t).load().expand_dims('time')
            smoothed_t = smooth(ds_t, kernel_radius_m=1700)
            values = smoothed_t['MaskConfidence'].values[0]
            mc_var[t, :, :] = np.clip(values / 0.01, 0, 100).astype(np.int8)
            del ds_t, smoothed_t, values

        f.attrs['pipeline'] = 'smoothed'

    ds.close()
    gc.collect()

    smoothed_ds = xr.open_dataset(save_path, chunks={'time': 1})
    tqdm.write(S.substep(f"Saved to {save_path}"))
    return smoothed_ds


def step_final(ds, fire_meta: dict, out_dir: str, calfire_gdf=None):
    """
    Final processing: round, binarize, trim, save netCDF, vectorize,
    save GeoJSON, and produce visualization.
    """
    from viz.gofer.fire_perimeter import plot_progression, plot_perimeter_comparison

    ### FIXME remove after diagnosis
    import psutil, sys                                                                
    rss_gb = psutil.Process().memory_info().rss / 1024**3                             
    tqdm.write(S.substep(f"RSS at step_final start: {rss_gb:.1f} GB", last_step=True))
    sys.stdout.flush()                                                                
    ###


    fire_name = fire_meta['fire_name']
    fire_year = fire_meta['fire_year']
    fire_id = f"{fire_name.lower()}_{fire_year}"

    tqdm.write(S.substep("Trimming, rounding, binarizing...", last_step=True))
    smoothed_path = str(Path(netcdf_dir) / 'smoothed.nc')
    first_fire, last_change = trim_inactive_timesteps(smoothed_path)

    ### FIXME remove after diagnosis
    rss_gb = psutil.Process().memory_info().rss / 1024**3
    tqdm.write(S.substep(f"RSS after trim: {rss_gb:.1f} GB", last_step=True))
    sys.stdout.flush()
    ###

    # Save final netCDF — slice-by-slice to avoid loading entire dataset
    import h5netcdf

    # Output subdirectories
    datasets_dir = Path(out_dir) / 'datasets'
    vectors_dir = Path(out_dir) / 'vectors'
    images_dir = Path(out_dir) / 'images'
    datasets_dir.mkdir(parents=True, exist_ok=True)
    vectors_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    nc_path = str(datasets_dir / f'{fire_id}_gofer.nc')
    trimmed_ds = ds.isel(time=slice(first_fire, last_change + 1))
    n_times = trimmed_ds.sizes['time']
    lat_vals = trimmed_ds.latitude.values
    lon_vals = trimmed_ds.longitude.values
    time_vals = trimmed_ds.time.values

    with h5netcdf.File(nc_path, 'w') as f:
        f.dimensions = {'time': n_times, 'latitude': len(lat_vals), 'longitude': len(lon_vals)}
        f.create_variable('time', ('time',), data=time_vals.astype('int64'))
        f.create_variable('latitude', ('latitude',), data=lat_vals.astype('float32'))
        f.create_variable('longitude', ('longitude',), data=lon_vals.astype('float32'))
        mc_var = f.create_variable(
            'MaskConfidence', ('time', 'latitude', 'longitude'),
            dtype='int8', fillvalue=np.int8(-1),
        )
        mc_var.attrs['scale_factor'] = np.float32(0.01)
        mc_var.attrs['add_offset'] = np.float32(0.0)

        for t in range(n_times):
            # Load, round, binarize one slice at a time
            slice_val = trimmed_ds['MaskConfidence'].isel(time=t).values
            slice_val = np.round(slice_val, 2)
            slice_val = np.where(slice_val < 0.95, 0.0, 1.0).astype(np.float32)
            mc_var[t, :, :] = np.clip(slice_val / 0.01, 0, 100).astype(np.int8)
            del slice_val

        # Attach metadata as global attrs
        f.attrs['pipeline'] = 'gofer_final'
        f.attrs['fire_name'] = fire_name
        f.attrs['fire_year'] = str(fire_year)
        f.attrs['fire_acres'] = str(fire_meta['fire_acres'])
        f.attrs['start_date'] = str(pd.Timestamp(time_vals[0]))
        f.attrs['end_date'] = str(pd.Timestamp(time_vals[-1]))
        f.attrs['lat_min'] = str(float(lat_vals.min()))
        f.attrs['lat_max'] = str(float(lat_vals.max()))
        f.attrs['lon_min'] = str(float(lon_vals.min()))
        f.attrs['lon_max'] = str(float(lon_vals.max()))

    ds.close()
    gc.collect()

    final_ds = xr.open_dataset(nc_path, chunks={'time': 1})
    tqdm.write(S.substep(f"Saved: {nc_path}", last_step=True))

    # Vectorize
    tqdm.write(S.substep("Vectorizing...", last_step=True))
    polygons = raster_to_polygon(final_ds, data_var='MaskConfidence', simplify_factor=2.0)

    # Save GeoJSON
    geojson_path = str(vectors_dir / f'{fire_id}_gofer.geojson')
    polygons.to_file(geojson_path, driver='GeoJSON')
    tqdm.write(S.substep(f"Saved: {geojson_path}", last_step=True))

    # Visualization
    tqdm.write(S.substep("Generating visualization...", last_step=True))
    viz_path = str(images_dir / f'{fire_id}_progression.png')
    plot_progression(
        gofer_gdf=polygons,
        ds=final_ds,
        title=f"GOFER {fire_name} {fire_year} — Fire Progression",
        save_path=viz_path,
    )
    tqdm.write(S.substep(f"Saved: {viz_path}", last_step=True))

    # Comparison visualization: GOFER vs FRAP final perimeters
    if calfire_gdf is not None:
        tqdm.write(S.substep("Generating comparison visualization...", last_step=True))
        comparison_path = str(images_dir / f'{fire_id}_comparison.png')
        plot_perimeter_comparison(
            gofer_gdf=polygons,
            ds=final_ds,
            calfire_gdf=calfire_gdf,
            title=f"GOFER vs FRAP — {fire_name} {fire_year}",
            save_path=comparison_path,
        )
    tqdm.write(S.substep(f"Saved: {comparison_path}", last_step=True))

    return final_ds, polygons


# --- Main ---

def main():
    args = parse_args()
    cfg = load_config(args.config)

    manifest = load_manifest(cfg['manifest'])
    tqdm.write(f"Config: {args.config}")
    tqdm.write(f"Manifest: {len(manifest)} fire(s)")
    tqdm.write(manifest.to_string(index=False))
    tqdm.write("")

    calfire_gdf = read_calfire_geojson(cfg['calfire_geojson'])

    if not args.skip_ingest:
        run_ingest(manifest, calfire_gdf, cfg)

    if not args.only_ingest:
        run_processing(manifest, calfire_gdf, cfg)

    tqdm.write("")
    tqdm.write(S.phase("Done."))


if __name__ == "__main__":
    main()
