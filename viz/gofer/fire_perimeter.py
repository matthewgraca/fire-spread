"""
Fire perimeter visualization.

Plots GOFER polygon perimeters with optional CalFire reference overlay
on a streetmap basemap.
"""
import geopandas as gpd
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
from matplotlib.colors import LinearSegmentedColormap

from viz.gofer.tilers import CartoDBTiles


def plot_perimeter(
    gofer_gdf: gpd.GeoDataFrame,
    ds: xr.Dataset = None,
    calfire_gdf: gpd.GeoDataFrame = None,
    extent: list = None,
    title: str = "GOFER Fire Perimeter",
    save_path: str = None,
    data_var: str = "MaskConfidence",
):
    """
    Plot GOFER polygon(s) with optional CalFire reference overlay on a
    streetmap basemap.

    If the GeoDataFrame has a 'time' column (multi-timestep), perimeters are
    colored from blue (early) to red (late), with early perimeters drawn on
    top so they are not obscured by later, larger perimeters.

    Color is normalized to the timestep at which the fire reaches 95% of its
    final burned area, matching the paper's visualization scheme.

    Args:
        gofer_gdf: GeoDataFrame from raster_to_polygon.
        ds: The source xarray Dataset, used to compute 95% area normalization.
        calfire_gdf: Optional CalFire reference perimeter GeoDataFrame.
        extent: [lon_min, lon_max, lat_min, lat_max] for the plot.
        title: Plot title.
        save_path: If provided, save the figure to this path.
        data_var: Name of the binary fire variable in ds.
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

    # Plot CalFire reference if provided (underneath everything)
    if calfire_gdf is not None:
        calfire_gdf.to_crs(epsg=4326).plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            facecolor='black',
            edgecolor='black',
            linewidth=2,
            label='CalFire',
            zorder=3,
        )

    # Plot GOFER polygon(s)
    has_time = 'time' in gofer_gdf.columns

    if has_time and len(gofer_gdf) > 1:
        n = len(gofer_gdf)
        # Custom colormap matching the GOFER paper's SpectralFancy palette
        colors = [
            (0.00, "#3089B4"),  # blue
            (0.10, "#59AAB2"),  # blue-cyan
            (0.20, "#92CCA9"),  # blue-green
            (0.30, "#BCE1AA"),  # light green
            (0.40, "#DCF4B7"),  # pale yellow-green
            (0.50, "#F7F3B3"),  # pale yellow
            (0.60, "#FDDD95"),  # light orange
            (0.70, "#FEBB73"),  # orange
            (0.80, "#F48E4F"),  # orange-red
            (0.90, "#EA5236"),  # red-orange
            (1.00, "#D7131A"),  # red
        ]
        cmap = LinearSegmentedColormap.from_list("fire_time", colors, N=256)
        cmap.set_over("#D91D1E")

        # Normalize color to 95% of final burned area
        if ds is not None and data_var in ds:
            fire_area = ds[data_var].sum(dim=['latitude', 'longitude'])
            final_area = float(fire_area.isel(time=-1))
            t95_idx = int((fire_area >= 0.95 * final_area).argmax(dim='time'))
            t95_idx = max(t95_idx, 1)  # avoid division by zero
        else:
            t95_idx = n - 1

        # If no calfire reference, draw the latest perimeter as a black background
        if calfire_gdf is None:
            gpd.GeoDataFrame([gofer_gdf.iloc[-1]], crs="EPSG:4326").plot(
                ax=ax,
                transform=ccrs.PlateCarree(),
                facecolor='black',
                edgecolor='black',
                alpha=1.0,
                linewidth=1.5,
                zorder=3,
            )

        # Draw late (large) perimeters first so early (small) ones sit on top
        for idx in reversed(range(n)):
            row = gofer_gdf.iloc[idx]
            frac = min(idx / t95_idx, 1.0)  # normalized to 95% area
            color = cmap(frac)
            gpd.GeoDataFrame([row], crs="EPSG:4326").plot(
                ax=ax,
                transform=ccrs.PlateCarree(),
                facecolor=color if calfire_gdf is not None else 'none',
                edgecolor=color,
                alpha=0.01 if calfire_gdf is not None else 0.8,
                linewidth=1.0,
                zorder=4 + (n - idx),  # earlier = higher zorder
            )

        # Colorbar
        norm = mcolors.Normalize(vmin=0, vmax=1)
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation='vertical', shrink=0.7, pad=0.03)

        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
        cbar.set_ticklabels(['0%', '25%', '50%', '75%', '95%+'])
        cbar.set_label("% of hours elapsed relative to 95% burned area")
    else:
        # Single polygon
        gofer_gdf.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            facecolor='red',
            edgecolor='red',
            alpha=0.35,
            linewidth=1.5,
            label='GOFER',
            zorder=4,
        )

    ax.set_title(title)
    if calfire_gdf is not None:
        ax.legend(loc='upper right')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.close()
