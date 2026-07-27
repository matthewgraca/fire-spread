"""
GOFER Ingest

Downloads GOES Fire Detection and Characterization data from NOAA's AWS
bucket for all fires in a manifest.

This is designed to run independently from the processing pipeline.
It is I/O-bound, idempotent (re-running skips existing files), and can
be left to run overnight.

Usage:
    python scripts/gofer/ingest.py --manifest manifests/example.csv
    python scripts/gofer/ingest.py --manifest manifests/example.csv --goes-dir data/goes

Manifest CSV format:
    state,year,fire_name
    CA,2020,BOBCAT
    CA,2020,CREEK
"""
from argparse import ArgumentParser
from pathlib import Path

from tqdm import tqdm

from gofer.ingest import download, read_calfire_geojson
from scripts.gofer.pipeline_helpers import (
    CALFIRE_GEOJSON, GOES_SAVE_DIR, TEMP_BASE_DIR,
    load_manifest, lookup_fire,
)


def parse_args():
    parser = ArgumentParser(description='GOFER Ingest — Download GOES data for fires in a manifest')
    parser.add_argument(
        '--manifest', type=str, required=True,
        help='Path to manifest CSV (columns: state, year, fire_name)'
    )
    parser.add_argument(
        '--goes-dir', type=str, default=GOES_SAVE_DIR,
        help='Directory for GOES data storage.'
    )
    parser.add_argument(
        '--temp-dir', type=str, default=TEMP_BASE_DIR,
        help='Base directory for per-fire temp/metadata storage.'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    manifest = load_manifest(args.manifest)
    tqdm.write(f"Manifest loaded: {len(manifest)} fire(s)")
    tqdm.write(manifest.to_string(index=False))
    tqdm.write("")

    calfire_gdf = read_calfire_geojson(CALFIRE_GEOJSON)

    for _, fire_row in tqdm(
        manifest.iterrows(),
        total=len(manifest),
        desc="Ingesting fires",
        unit="fire",
    ):
        fire_name = fire_row['fire_name']
        fire_year = int(fire_row['year'])
        fire_id = f"{fire_name.lower()}_{fire_year}"
        temp_dir = str(Path(args.temp_dir) / fire_id)

        tqdm.write(f"\n{'='*60}")
        tqdm.write(f"Ingesting: {fire_name} ({fire_year})")
        tqdm.write(f"{'='*60}")

        try:
            fire = lookup_fire(calfire_gdf, fire_name, fire_year)

            download(
                start=fire['ALARM_DATE'],
                end=fire['CONT_DATE'],
                fire_name=str(fire['FIRE_NAME']),
                fire_year=int(fire['YEAR_']),
                fire_acres=int(fire['GIS_ACRES']),
                goes_save_dir=args.goes_dir,
                metadata_save_dir=temp_dir,
                subhourly=True,
                lon_min=float(fire['bbox_min_lon']),
                lon_max=float(fire['bbox_max_lon']),
                lat_min=float(fire['bbox_min_lat']),
                lat_max=float(fire['bbox_max_lat']),
            )
            tqdm.write(f"  ✓ {fire_name} ({fire_year}) ingested.")
        except Exception as e:
            tqdm.write(f"  ✗ {fire_name} ({fire_year}) FAILED: {e}")
            import traceback
            tqdm.write(traceback.format_exc())
            continue

    tqdm.write(f"\nIngest finished for {len(manifest)} fire(s).")


if __name__ == "__main__":
    main()
