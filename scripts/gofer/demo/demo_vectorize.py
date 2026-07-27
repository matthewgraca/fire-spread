"""
Demo: vectorize a GOFER output netCDF and plot the resulting polygon(s)
against the CalFire reference perimeter, with a streetmap basemap.

Supports multi-timestep datasets, plotting early perimeters in blue
through late perimeters in red, with early perimeters drawn on top.
Color is normalized to 95% of total burned area, matching the paper.
"""
import xarray as xr
import geopandas as gpd
from gofer.vectorize import raster_to_polygon
from viz.gofer.fire_perimeter import plot_perimeter


if __name__ == "__main__":
    # Load the GOFER output
    ds = xr.open_dataset('out/bobcat_2020_gofer.nc')
    print(ds)

    # Vectorize
    gofer_gdf = raster_to_polygon(ds)
    print(gofer_gdf)
    print(f"Simplification applied with factor=2.0")
    print(f"Number of perimeters: {len(gofer_gdf)}")

    # Load CalFire reference
    calfire = gpd.read_file(
        "data/calfire/California_Historic_Fire_Perimeters_-4891938132824355098.geojson"
    )
    bobcat_ref = calfire.loc[calfire['FIRE_NAME'] == 'BOBCAT'].to_crs(epsg=4326)

    # Derive extent from the dataset
    buffer = 0.05
    extent = [
        float(ds.longitude.min()) - buffer,
        float(ds.longitude.max()) + buffer,
        float(ds.latitude.min()) - buffer,
        float(ds.latitude.max()) + buffer,
    ]

    # Plot
    plot_perimeter(
        gofer_gdf=gofer_gdf,
        ds=ds,
        calfire_gdf=bobcat_ref,
        extent=extent,
        title="GOFER Bobcat 2020 — Fire Progression",
        save_path="out/gofer/bobcat_polygon.png",
    )

    plot_perimeter(
        gofer_gdf=gofer_gdf,
        ds=ds,
        calfire_gdf=None,
        extent=extent,
        title="GOFER Bobcat 2020 — Fire Progression",
        save_path="out/gofer/bobcat_polygon_paper.png",
    )
