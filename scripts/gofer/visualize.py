"""
GOFER Fire Perimeter Visualization Script

Generates fire perimeter visualizations from pipeline outputs.

Usage:
    python scripts/gofer/visualize.py edge-progression --dataset fire.nc --vectors fire.geojson -o output.png
    python scripts/gofer/visualize.py filled-progression --dataset fire.nc --vectors fire.geojson --calfire ref.geojson -o output.png
    python scripts/gofer/visualize.py final-perimeter --dataset fire.nc --vectors fire.geojson --calfire ref.geojson -o output.png
"""
from argparse import ArgumentParser

import geopandas as gpd
import xarray as xr

from viz.gofer.fire_perimeter import (
    plot_progression,
    plot_progression_filled,
    plot_perimeter_comparison,
)


def _load_calfire(calfire_path: str, ds: xr.Dataset) -> gpd.GeoDataFrame:
    """Load and filter CalFire GeoJSON to the fire matching the dataset metadata."""
    fire_name = ds.attrs.get('fire_name')
    fire_year = ds.attrs.get('fire_year')

    gdf = gpd.read_file(calfire_path)

    if fire_name is None or fire_year is None:
        raise ValueError(
            "Dataset is missing 'fire_name' or 'fire_year' attributes. "
            "Cannot determine which fire to filter from CalFire data."
        )

    match = gdf.loc[
        (gdf['FIRE_NAME'] == fire_name) & (gdf['YEAR_'] == int(fire_year))
    ]

    if len(match) == 0:
        raise ValueError(
            f"Fire '{fire_name}' ({fire_year}) not found in CalFire data at {calfire_path}."
        )

    return match.to_crs(epsg=4326)


def parse_args():
    parser = ArgumentParser(
        description="Generate GOFER fire perimeter visualizations."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Shared arguments
    def add_common_args(sub):
        sub.add_argument(
            "--dataset", type=str, required=True,
            help="Path to the GOFER netCDF dataset."
        )
        sub.add_argument(
            "--vectors", type=str, required=True,
            help="Path to the GOFER vectorized GeoJSON."
        )
        sub.add_argument(
            "-o", "--output", type=str, required=True,
            help="Output image path (e.g., output.png)."
        )
        sub.add_argument(
            "--title", type=str, default=None,
            help="Plot title. Auto-generated if not provided."
        )

    # edge-progression
    edge = subparsers.add_parser(
        "edge-progression",
        help="Colored edge-only perimeter progression (paper style)."
    )
    add_common_args(edge)

    # filled-progression
    filled = subparsers.add_parser(
        "filled-progression",
        help="Filled facecolor perimeter progression."
    )
    add_common_args(filled)
    filled.add_argument(
        "--calfire", type=str, default=None,
        help="Path to CalFire/FRAP reference GeoJSON (optional)."
    )

    # final-perimeter
    comparison = subparsers.add_parser(
        "final-perimeter",
        help="GOFER vs FRAP final perimeter comparison."
    )
    add_common_args(comparison)
    comparison.add_argument(
        "--calfire", type=str, required=True,
        help="Path to CalFire/FRAP reference GeoJSON."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    ds = xr.open_dataset(args.dataset, chunks='auto')
    gofer_gdf = gpd.read_file(args.vectors)

    title = args.title

    if args.mode == "edge-progression":
        if title is None:
            title = "GOFER Fire Progression"
        plot_progression(
            gofer_gdf=gofer_gdf,
            ds=ds,
            title=title,
            save_path=args.output,
        )

    elif args.mode == "filled-progression":
        calfire_gdf = None
        if args.calfire:
            calfire_gdf = _load_calfire(args.calfire, ds)
        if title is None:
            title = "GOFER Fire Progression (Filled)"
        plot_progression_filled(
            gofer_gdf=gofer_gdf,
            ds=ds,
            calfire_gdf=calfire_gdf,
            title=title,
            save_path=args.output,
        )

    elif args.mode == "final-perimeter":
        calfire_gdf = _load_calfire(args.calfire, ds)
        if title is None:
            title = "GOFER vs FRAP — Final Perimeter Comparison"
        plot_perimeter_comparison(
            gofer_gdf=gofer_gdf,
            ds=ds,
            calfire_gdf=calfire_gdf,
            title=title,
            save_path=args.output,
        )

    ds.close()


if __name__ == "__main__":
    main()
