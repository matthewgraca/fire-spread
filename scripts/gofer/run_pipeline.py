"""
GOFER Processing Pipeline

Processes one or more fires from a manifest file through the GOFER pipeline,
producing fire perimeter netCDF files, GeoJSON polygons, and progression
visualizations.

Assumes GOES data has already been ingested via ingest.py.

Usage:
    python scripts/gofer/run_pipeline.py --manifest manifests/example.csv --step aggregate
    python scripts/gofer/run_pipeline.py --manifest manifests/example.csv --step final
    python scripts/gofer/run_pipeline.py --manifest manifests/example.csv --step aggregate --clean

Manifest CSV format:
    state,year,fire_name
    CA,2020,BOBCAT
    CA,2020,CREEK
"""
import pickle
import shutil
from argparse import ArgumentParser
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from gofer.composite import composite
from gofer.early_perimeter_adjustment import apply_scaling_factors, get_scaling_factors
from gofer.goes_utils import eval_and_save_nc
from gofer.ingest import read_calfire_geojson
from gofer.ortho import orthorectify
from gofer.postprocess import binarize, round_to, trim_inactive_timesteps
from gofer.spatial_smoothing import smooth
from gofer.temporal_downsampler import aggregate
from gofer.vectorize import raster_to_polygon

from scripts.gofer.pipeline_helpers import (
    CALFIRE_GEOJSON, GOES_SAVE_DIR, DEM_FILEPATH, TEMP_BASE_DIR, BBOX_BUFFER,
    load_manifest, lookup_fire,
)


# --- Configuration ---

PIPELINE_STEPS = [
    'aggregate',
    'scale',
    'ortho',
    'composite',
    'smooth',
    'final',
]


# --- Argument parsing ---

def parse_args():
    parser = ArgumentParser(description='GOFER Processing Pipeline')
    parser.add_argument(
        '--manifest', type=str, required=True,
        help='Path to manifest CSV (columns: state, year, fire_name)'
    )
    parser.add_argument(
        '--step', type=str, required=True, choices=PIPELINE_STEPS,
        help='Pipeline step to start from. All subsequent steps will also run.'
    )
    parser.add_argument(
        '--clean', action='store_true',
        help='Delete intermediate netCDF temp files after completion.'
    )
    parser.add_argument(
        '--dem', type=str, default=DEM_FILEPATH,
        help='Path to the DEM GeoTIFF.'
    )
    parser.add_argument(
        '--goes-dir', type=str, default=GOES_SAVE_DIR,
        help='Directory for GOES data storage.'
    )
    parser.add_argument(
        '--temp-dir', type=str, default=TEMP_BASE_DIR,
        help='Base directory for per-fire temp/metadata storage.'
    )
    parser.add_argument(
        '--out-dir', type=str, default='out',
        help='Output directory for final products.'
    )
    return parser.parse_args()


def get_active_steps(start_step: str) -> list[bool]:
    """Determine which pipeline steps are active based on start step."""
    start_idx = PIPELINE_STEPS.index(start_step)
    return [i >= start_idx for i in range(len(PIPELINE_STEPS))]


# --- Pipeline steps ---

def step_aggregate(goes_save_dir: str, temp_dir: str, netcdf_dir: str,
                   dates: pd.DatetimeIndex, fire_name: str):
    """Remap, temporally downsample, and aggregate both satellites."""
    results = {}
    for sat in ['west', 'east']:
        tqdm.write(f"    Aggregating GOES-{sat.capitalize()}...")
        ds = aggregate(
            goes_save_dir=goes_save_dir,
            csv_path=str(Path(temp_dir) / f'{sat}_files.csv'),
            temp_dir=str(Path(netcdf_dir) / sat / 'hourly'),
            dates=dates,
            fire_name=fire_name,
            verbose=False,
        )
        ds = eval_and_save_nc(
            ds,
            chunk_size=(1, 1500, 2500),
            save_path=str(Path(netcdf_dir) / sat / 'aggregated.nc'),
            chunks='auto',
            desc=f'{sat} aggregation',
            verbose=False,
        )
        results[sat] = ds
    return results['west'], results['east']


def step_scale(west_ds, east_ds, dem_filepath: str, bbox: tuple, netcdf_dir: str):
    """Compute and apply early perimeter scaling factors."""
    results = {}
    for sat, ds in [('west', west_ds), ('east', east_ds)]:
        tqdm.write(f"    Scaling GOES-{sat.capitalize()}...")
        sf = get_scaling_factors(
            ds,
            ortho_kwargs={'dem_filepath': dem_filepath, 'bbox': bbox},
            show_progress=False,
        )
        scaled_ds = apply_scaling_factors(ds, sf)
        ds.close()
        scaled_ds = eval_and_save_nc(
            scaled_ds,
            chunk_size=(1, 1500, 2500),
            save_path=str(Path(netcdf_dir) / sat / 'scaled.nc'),
            chunks={'time': 1},
            desc=f'{sat} scaling',
            verbose=False,
        )
        results[sat] = scaled_ds
    return results['west'], results['east']


def step_ortho(west_ds, east_ds, dem_filepath: str, bbox: tuple, netcdf_dir: str):
    """Orthorectify both satellite datasets."""
    results = {}
    for sat, ds in [('west', west_ds), ('east', east_ds)]:
        tqdm.write(f"    Orthorectifying GOES-{sat.capitalize()}...")
        ortho_ds = orthorectify(
            ds,
            dem_filepath=dem_filepath,
            bbox=bbox,
            data_var="MaskConfidence",
        )
        ds.close()
        ortho_ds = eval_and_save_nc(
            ortho_ds,
            save_path=str(Path(netcdf_dir) / sat / 'ortho.nc'),
            chunks='auto',
            desc=f'{sat} orthorectification',
            verbose=False,
        )
        results[sat] = ortho_ds
    return results['west'], results['east']


def step_composite(west_ds, east_ds, dates: pd.DatetimeIndex, netcdf_dir: str):
    """Composite East and West into a single dataset."""
    tqdm.write("    Compositing East and West...")
    composite_ds = composite(west_ds, east_ds, dates, data_var='MaskConfidence')
    west_ds.close()
    east_ds.close()
    composite_ds = eval_and_save_nc(
        composite_ds,
        save_path=str(Path(netcdf_dir) / 'composited.nc'),
        chunks='auto',
        desc='compositing',
        verbose=False,
    )
    return composite_ds


def step_smooth(ds, netcdf_dir: str):
    """Apply spatial smoothing."""
    tqdm.write("    Smoothing...")
    smoothed_ds = smooth(ds, kernel_radius_m=1700)
    ds.close()
    smoothed_ds = eval_and_save_nc(
        smoothed_ds,
        save_path=str(Path(netcdf_dir) / 'smoothed.nc'),
        chunks='auto',
        desc='smoothing',
        verbose=False,
    )
    return smoothed_ds


def step_final(ds, fire_meta: dict, out_dir: str, calfire_gdf=None):
    """
    Final processing: round, binarize, trim, save netCDF, vectorize,
    save GeoJSON, and produce visualization.
    """
    from scripts.gofer.demo.demo_vectorize import plot_perimeter

    fire_name = fire_meta['fire_name']
    fire_year = fire_meta['fire_year']
    fire_id = f"{fire_name.lower()}_{fire_year}"

    tqdm.write("    Rounding, binarizing, trimming...")
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
        verbose=False,
    )
    tqdm.write(f"    Saved: {nc_path}")

    # Vectorize
    tqdm.write("    Vectorizing...")
    polygons = raster_to_polygon(final_ds, data_var='MaskConfidence', simplify_factor=2.0)

    # Save GeoJSON
    geojson_path = str(vectors_dir / f'{fire_id}_gofer.geojson')
    polygons.to_file(geojson_path, driver='GeoJSON')
    tqdm.write(f"    Saved: {geojson_path}")

    # Visualization
    tqdm.write("    Generating visualization...")
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
        calfire_gdf=calfire_gdf,
        extent=extent,
        title=f"GOFER {fire_name} {fire_year} — Fire Progression",
        save_path=viz_path,
    )

    return final_ds, polygons


# --- Main ---

def process_fire(
    fire_row: pd.Series,
    calfire_gdf: gpd.GeoDataFrame,
    active_steps: list[bool],
    args,
):
    """Run the pipeline for a single fire."""
    fire_name = fire_row['fire_name']
    fire_year = int(fire_row['year'])
    fire_id = f"{fire_name.lower()}_{fire_year}"

    fire = lookup_fire(calfire_gdf, fire_name, fire_year)

    temp_dir = str(Path(args.temp_dir) / fire_id)
    netcdf_dir = f'{temp_dir}/netcdf'
    out_dir = args.out_dir

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    fire_meta = {
        'fire_name': fire_name,
        'fire_year': fire_year,
        'fire_acres': int(fire['GIS_ACRES']),
    }

    # Verify ingest was run
    metadata_path = Path(temp_dir) / 'metadata.pkl'
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata not found at {metadata_path}. "
            f"Run ingest.py first for {fire_name} ({fire_year})."
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

    step_idx = 0

    # [1/6] Aggregate
    if active_steps[step_idx]:
        tqdm.write(f"  [1/6] Aggregating...")
        west_ds, east_ds = step_aggregate(
            args.goes_dir, temp_dir, netcdf_dir, dates, fire_name
        )
    else:
        west_ds = xr.open_dataset(f'{netcdf_dir}/west/aggregated.nc', chunks='auto')
        east_ds = xr.open_dataset(f'{netcdf_dir}/east/aggregated.nc', chunks='auto')
    step_idx += 1

    # [2/6] Scale
    if active_steps[step_idx]:
        tqdm.write(f"  [2/6] Scaling early perimeters...")
        west_ds, east_ds = step_scale(west_ds, east_ds, args.dem, bbox, netcdf_dir)
    else:
        west_ds = xr.open_dataset(f'{netcdf_dir}/west/scaled.nc', chunks='auto')
        east_ds = xr.open_dataset(f'{netcdf_dir}/east/scaled.nc', chunks='auto')
    step_idx += 1

    # [3/6] Ortho
    if active_steps[step_idx]:
        tqdm.write(f"  [3/6] Orthorectifying...")
        west_ds, east_ds = step_ortho(west_ds, east_ds, args.dem, bbox, netcdf_dir)
    else:
        west_ds = xr.open_dataset(f'{netcdf_dir}/west/ortho.nc', chunks='auto')
        east_ds = xr.open_dataset(f'{netcdf_dir}/east/ortho.nc', chunks='auto')
    step_idx += 1

    # [4/6] Composite
    if active_steps[step_idx]:
        tqdm.write(f"  [4/6] Compositing...")
        ds = step_composite(west_ds, east_ds, dates, netcdf_dir)
    else:
        ds = xr.open_dataset(f'{netcdf_dir}/composited.nc', chunks='auto')
    step_idx += 1

    # [5/6] Smooth
    if active_steps[step_idx]:
        tqdm.write(f"  [5/6] Smoothing...")
        ds = step_smooth(ds, netcdf_dir)
    else:
        ds = xr.open_dataset(f'{netcdf_dir}/smoothed.nc', chunks='auto')
    step_idx += 1

    # [6/6] Final
    if active_steps[step_idx]:
        tqdm.write(f"  [6/6] Final processing...")
        calfire_ref = calfire_gdf.loc[
            (calfire_gdf['FIRE_NAME'] == fire_name) &
            (calfire_gdf['YEAR_'] == fire_year)
        ].to_crs(epsg=4326)
        calfire_ref = calfire_ref if len(calfire_ref) > 0 else None

        step_final(ds, fire_meta, out_dir, calfire_gdf=calfire_ref)

    # Cleanup intermediates
    if args.clean and Path(netcdf_dir).exists():
        tqdm.write(f"  Cleaning intermediate files...")
        shutil.rmtree(netcdf_dir)


def main():
    args = parse_args()

    manifest = load_manifest(args.manifest)
    tqdm.write(f"Manifest loaded: {len(manifest)} fire(s)")
    tqdm.write(manifest.to_string(index=False))
    tqdm.write("")

    active_steps = get_active_steps(args.step)
    active_names = [s for s, a in zip(PIPELINE_STEPS, active_steps) if a]
    tqdm.write(f"Pipeline: {' → '.join(active_names)}")
    tqdm.write("")

    calfire_gdf = read_calfire_geojson(CALFIRE_GEOJSON)

    for _, fire_row in tqdm(
        manifest.iterrows(),
        total=len(manifest),
        desc="Processing fires",
        unit="fire",
    ):
        fire_name = fire_row['fire_name']
        fire_year = fire_row['year']
        tqdm.write(f"\n{'='*60}")
        tqdm.write(f"Processing: {fire_name} ({fire_year})")
        tqdm.write(f"{'='*60}")

        try:
            process_fire(fire_row, calfire_gdf, active_steps, args)
            tqdm.write(f"  ✓ {fire_name} ({fire_year}) complete.")
        except Exception as e:
            tqdm.write(f"  ✗ {fire_name} ({fire_year}) FAILED: {e}")
            import traceback
            tqdm.write(traceback.format_exc())
            continue

    tqdm.write(f"\nPipeline finished for {len(manifest)} fire(s).")


if __name__ == "__main__":
    main()
