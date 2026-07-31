"""
Demo: vectorize a GOFER output netCDF and plot the resulting polygon(s).

Supports multi-timestep datasets, plotting early perimeters in blue
through late perimeters in red, with early perimeters drawn on top.
Color is normalized to 95% of total burned area, matching the paper.
"""
import xarray as xr
import geopandas as gpd
from gofer.vectorize import raster_to_polygon
from viz.gofer.fire_perimeter import plot_progression, plot_perimeter_comparison


if __name__ == "__main__":
    # Load the GOFER output
    ds = xr.open_dataset('out/bobcat_2020_gofer.nc')
    print(ds)

    # Vectorize
    gofer_gdf = raster_to_polygon(ds, data_var='MaskConfidence', simplify_factor=2.0)
    print(gofer_gdf)
    print(f"Number of perimeters: {len(gofer_gdf)}")

    # Load CalFire reference
    calfire = gpd.read_file(
        "data/calfire/California_Historic_Fire_Perimeters_-4891938132824355098.geojson"
    )
    bobcat_ref = calfire.loc[calfire['FIRE_NAME'] == 'BOBCAT'].to_crs(epsg=4326)

    # Plot progression (colored edges, paper style)
    plot_progression(
        gofer_gdf=gofer_gdf,
        ds=ds,
        title="GOFER Bobcat 2020 — Fire Progression",
        save_path="out/gofer/bobcat_progression.png",
    )

    # Plot comparison against CalFire reference perimeter
    plot_perimeter_comparison(
        gofer_gdf=gofer_gdf,
        ds=ds,
        calfire_gdf=bobcat_ref,
        title="GOFER vs FRAP — Bobcat 2020",
        save_path="out/gofer/bobcat_comparison.png",
    )
