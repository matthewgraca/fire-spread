"""
Demo: vectorize a GOFER output netCDF and plot the resulting polygon
against the CalFire reference perimeter, with a streetmap basemap.
"""
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from gofer.vectorize import raster_to_polygon
from viz.gofer.tilers import CartoDBTiles


def plot_perimeter(
    gofer_gdf: gpd.GeoDataFrame,
    calfire_gdf: gpd.GeoDataFrame = None,
    extent: list = None,
    title: str = "GOFER Fire Perimeter",
    save_path: str = None,
):
    """
    Plot the GOFER polygon with optional CalFire reference overlay on a 
    streetmap basemap.

    Args:
        gofer_gdf: GeoDataFrame from raster_to_polygon.
        calfire_gdf: Optional CalFire reference perimeter GeoDataFrame.
        extent: [lon_min, lon_max, lat_min, lat_max] for the plot.
        title: Plot title.
        save_path: If provided, save the figure to this path.
    """
    tiler = CartoDBTiles(style='rastertiles/voyager', cache=True)
    fig, ax = plt.subplots(
        1, 1, figsize=(16, 12),
        subplot_kw={'projection': ccrs.PlateCarree()},
        layout='constrained'
    )

    # Streetmap basemap
    ax.add_image(tiler, 12)

    if extent is not None:
        ax.set_extent(extent, crs=ccrs.PlateCarree())

    # Plot GOFER polygon
    gofer_gdf.to_crs(epsg=4326).plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        facecolor='red',
        edgecolor='red',
        alpha=0.35,
        linewidth=1.5,
        label='GOFER'
    )

    # Plot CalFire reference if provided
    if calfire_gdf is not None:
        calfire_gdf.to_crs(epsg=4326).plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            facecolor='none',
            edgecolor='black',
            linewidth=2,
            label='CalFire'
        )

    ax.set_title(title)
    ax.legend(loc='upper right')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()
    plt.close()


if __name__ == "__main__":
    # Load the GOFER output
    ds = xr.open_dataset('out/bobcat_2020_gofer.nc')
    print(ds)

    # Vectorize
    factor = 2.0
    gofer_gdf = raster_to_polygon(ds, simplify_factor=factor)
    print(gofer_gdf)
    gofer_gdf.to_file('temp/bobcat_2020/bobcat_polygons.geojson', driver='GeoJSON')
    print(f"Simplification applied with factor={factor}")

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
        calfire_gdf=bobcat_ref,
        extent=extent,
        title="GOFER Bobcat 2020 — Vectorized Perimeter",
        save_path="out/gofer/bobcat_polygon.png",
    )
