"""
GOFER Pipeline — Unified Runner

Runs the full GOFER pipeline (ingest + processing) for all fires in a
manifest, configured via a YAML file.

Usage:
    python scripts/gofer/run.py --config configs/gofer.yaml
    python scripts/gofer/run.py --config configs/gofer.yaml --skip-ingest
    python scripts/gofer/run.py --config configs/gofer.yaml --only-ingest
"""
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
        fire_id = f"{fire_name.lower()}_{fire_year}"
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

    # [1/6] Aggregate
    tqdm.write(S.step(1, 6, "Aggregating..."))
    west_ds, east_ds = step_aggregate(cfg['goes_dir'], temp_dir, netcdf_dir, dates, fire_name, cfg['workers'])

    # [2/6] Scale
    tqdm.write(S.step(2, 6, "Scaling early perimeters..."))
    west_ds, east_ds = step_scale(west_ds, east_ds, dem, bbox, netcdf_dir)

    # [3/6] Ortho
    tqdm.write(S.step(3, 6, "Orthorectifying..."))
    west_ds, east_ds = step_ortho(west_ds, east_ds, dem, bbox, netcdf_dir)

    # [4/6] Composite
    tqdm.write(S.step(4, 6, "Compositing..."))
    ds = step_composite(west_ds, east_ds, dates, netcdf_dir)

    # [5/6] Smooth
    tqdm.write(S.step(5, 6, "Smoothing..."))
    ds = step_smooth(ds, netcdf_dir)

    # [6/6] Final
    tqdm.write(S.step(6, 6, "Final processing..."))
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
    return results['west'], results['east']


def step_ortho(west_ds, east_ds, dem_filepath: str, bbox: tuple, netcdf_dir: str):
    """Orthorectify both satellite datasets."""
    results = {}
    for sat, ds in [('west', west_ds), ('east', east_ds)]:
        tqdm.write(S.substep(f"Orthorectifying GOES-{sat.capitalize()}..."))
        save_path = str(Path(netcdf_dir) / sat / 'ortho.nc')
        ortho_ds = orthorectify(
            ds,
            dem_filepath=dem_filepath,
            bbox=bbox,
            data_var="MaskConfidence",
        )
        ds.close()
        ortho_ds = eval_and_save_nc(
            ortho_ds,
            save_path=save_path,
            chunks={'time': 1},
            desc=f'{sat} orthorectification',
            verbose=True,
        )
        tqdm.write(S.substep(f"Saved to {save_path}"))
        results[sat] = ortho_ds
    return results['west'], results['east']


def step_composite(west_ds, east_ds, dates: pd.DatetimeIndex, netcdf_dir: str):
    """Composite East and West into a single dataset."""
    import dask

    tqdm.write(S.substep("Compositing East and West..."))
    save_path = str(Path(netcdf_dir) / 'composited.nc')

    with dask.config.set(scheduler='synchronous'):
        composite_ds = composite(west_ds, east_ds, dates, data_var='MaskConfidence')
        west_ds.close()
        east_ds.close()
        composite_ds = eval_and_save_nc(
            composite_ds,
            save_path=save_path,
            chunks={'time': 1},
            desc='compositing',
            verbose=True,
        )
    tqdm.write(S.substep(f"Saved to {save_path}"))
    return composite_ds


def step_smooth(ds, netcdf_dir: str):
    """Apply spatial smoothing."""
    tqdm.write(S.substep("Smoothing..."))
    save_path = str(Path(netcdf_dir) / 'smoothed.nc')
    smoothed_ds = smooth(ds, kernel_radius_m=1700)
    ds.close()
    smoothed_ds = eval_and_save_nc(
        smoothed_ds,
        save_path=save_path,
        chunks='auto',
        desc='smoothing',
        verbose=True,
    )
    tqdm.write(S.substep(f"Saved to {save_path}"))
    return smoothed_ds


def step_final(ds, fire_meta: dict, out_dir: str, calfire_gdf=None):
    """
    Final processing: round, binarize, trim, save netCDF, vectorize,
    save GeoJSON, and produce visualization.
    """
    from viz.gofer.fire_perimeter import plot_perimeter

    fire_name = fire_meta['fire_name']
    fire_year = fire_meta['fire_year']
    fire_id = f"{fire_name.lower()}_{fire_year}"

    tqdm.write(S.substep("Rounding, binarizing, trimming...", last_step=True))
    final_ds = round_to(ds, data_var='MaskConfidence', decimals=2)
    final_ds = binarize(final_ds, data_var='MaskConfidence', threshold=0.95)
    final_ds = trim_inactive_timesteps(final_ds, data_var='MaskConfidence')

    # Attach metadata
    final_ds = final_ds.assign_attrs(
        pipeline='gofer_final',
        fire_name=fire_name,
        fire_year=fire_year,
        fire_acres=fire_meta['fire_acres'],
        start_date=str(pd.Timestamp(final_ds.time.values[0])),
        end_date=str(pd.Timestamp(final_ds.time.values[-1])),
        lat_min=float(final_ds.latitude.min()),
        lat_max=float(final_ds.latitude.max()),
        lon_min=float(final_ds.longitude.min()),
        lon_max=float(final_ds.longitude.max()),
    )

    # Output subdirectories
    datasets_dir = Path(out_dir) / 'datasets'
    vectors_dir = Path(out_dir) / 'vectors'
    images_dir = Path(out_dir) / 'images'
    datasets_dir.mkdir(parents=True, exist_ok=True)
    vectors_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    # Save final netCDF
    nc_path = str(datasets_dir / f'{fire_id}_gofer.nc')
    final_ds = eval_and_save_nc(
        final_ds,
        save_path=nc_path,
        chunks='auto',
        desc='final processing',
        verbose=True,
    )
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
    buffer = 0.05
    extent = [
        float(final_ds.longitude.min()) - buffer,
        float(final_ds.longitude.max()) + buffer,
        float(final_ds.latitude.min()) - buffer,
        float(final_ds.latitude.max()) + buffer,
    ]

    viz_path = str(images_dir / f'{fire_id}_progression.png')
    plot_perimeter(
        gofer_gdf=polygons,
        ds=final_ds,
        #calfire_gdf=calfire_gdf,   # gee version (colored facecolors)
        calfire_gdf=None,           # paper version (edges, black facecolor)
        extent=extent,
        title=f"GOFER {fire_name} {fire_year} — Fire Progression",
        save_path=viz_path,
    )

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
