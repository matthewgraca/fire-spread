"""
Fire perimeter visualization.

Provides three visualization functions:
- plot_progression: Edge-only perimeter progression (paper style)
- plot_progression_filled: Filled facecolor perimeter progression
- plot_perimeter_comparison: GOFER vs FRAP final perimeter comparison
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

import warnings
warnings.filterwarnings("ignore", message="Cartopy created the following directory to cache")


# --- Shared utilities ---

_FIRE_CMAP_COLORS = [
    (0.00, "#3089B4"),
    (0.10, "#59AAB2"),
    (0.20, "#92CCA9"),
    (0.30, "#BCE1AA"),
    (0.40, "#DCF4B7"),
    (0.50, "#F7F3B3"),
    (0.60, "#FDDD95"),
    (0.70, "#FEBB73"),
    (0.80, "#F48E4F"),
    (0.90, "#EA5236"),
    (1.00, "#D7131A"),
]


def _fire_cmap():
    cmap = LinearSegmentedColormap.from_list("fire_time", _FIRE_CMAP_COLORS, N=256)
    cmap.set_over("#D91D1E")
    return cmap


def _extent_from_ds(ds: xr.Dataset, buffer: float = 0.05) -> list:
    return [
        float(ds.longitude.min()) - buffer,
        float(ds.longitude.max()) + buffer,
        float(ds.latitude.min()) - buffer,
        float(ds.latitude.max()) + buffer,
    ]


def _setup_basemap(extent: list):
    tiler = CartoDBTiles(style='rastertiles/voyager', cache=True)
    fig, ax = plt.subplots(
        1, 1, figsize=(16, 12),
        subplot_kw={'projection': ccrs.PlateCarree()},
        layout='constrained'
    )
    ax.add_image(tiler, 12)
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    return fig, ax


def _t95_index(ds: xr.Dataset, data_var: str, n: int) -> int:
    """Find the timestep index at which 95% of final burned area is reached."""
    if ds is not None and data_var in ds:
        fire_area = ds[data_var].sum(dim=['latitude', 'longitude'])
        final_area = float(fire_area.isel(time=-1))
        t95_idx = int((fire_area >= 0.95 * final_area).argmax(dim='time'))
        return max(t95_idx, 1)
    return n - 1


def _add_colorbar(fig, ax, cmap):
    norm = mcolors.Normalize(vmin=0, vmax=1)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', shrink=0.7, pad=0.03)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(['0%', '25%', '50%', '75%', '95%+'])
    cbar.set_label("% of hours elapsed relative to 95% burned area")


# --- Public API ---

def plot_progression(
    gofer_gdf: gpd.GeoDataFrame,
    ds: xr.Dataset,
    title: str = "GOFER Fire Progression",
    save_path: str = None,
    data_var: str = "MaskConfidence",
    step: int = 1,
):
    """
    Plot GOFER perimeter progression with colored edges on a black
    final-perimeter background (paper style).

    Perimeters are colored from blue (early) to red (late), with early
    perimeters drawn on top. Color is normalized to the timestep at which
    the fire reaches 95% of its final burned area.

    Args:
        gofer_gdf: GeoDataFrame from raster_to_polygon (multi-timestep).
        ds: The source xarray Dataset for extent and 95% normalization.
        title: Plot title.
        save_path: If provided, save the figure to this path.
        data_var: Name of the binary fire variable in ds.
        step: Plot every Nth timestep. 1 = all, 12 = every 12th.
    """
    extent = _extent_from_ds(ds)
    fig, ax = _setup_basemap(extent)
    cmap = _fire_cmap()

    n = len(gofer_gdf)
    t95_idx = _t95_index(ds, data_var, n)

    # Subsample indices
    indices = list(range(0, n, step))
    if indices[-1] != n - 1:
        indices.append(n - 1)  # always include final perimeter

    # Black background: final perimeter
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
    for idx in reversed(indices):
        row = gofer_gdf.iloc[idx]
        frac = min(idx / t95_idx, 1.0)
        color = cmap(frac)
        gpd.GeoDataFrame([row], crs="EPSG:4326").plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            facecolor='none',
            edgecolor=color,
            alpha=0.8,
            linewidth=1.0,
            zorder=4 + (n - idx),
        )

    _add_colorbar(fig, ax, cmap)
    ax.set_title(title)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.close()


def plot_progression_filled(
    gofer_gdf: gpd.GeoDataFrame,
    ds: xr.Dataset,
    calfire_gdf: gpd.GeoDataFrame = None,
    title: str = "GOFER Fire Progression",
    save_path: str = None,
    data_var: str = "MaskConfidence",
):
    """
    Plot GOFER perimeter progression with filled facecolors.

    Perimeters are colored from blue (early) to red (late) with filled
    faces at low alpha. Optionally overlays a CalFire reference perimeter
    underneath.

    Args:
        gofer_gdf: GeoDataFrame from raster_to_polygon (multi-timestep).
        ds: The source xarray Dataset for extent and 95% normalization.
        calfire_gdf: Optional CalFire reference perimeter GeoDataFrame.
        title: Plot title.
        save_path: If provided, save the figure to this path.
        data_var: Name of the binary fire variable in ds.
    """
    extent = _extent_from_ds(ds)
    fig, ax = _setup_basemap(extent)
    cmap = _fire_cmap()

    n = len(gofer_gdf)
    t95_idx = _t95_index(ds, data_var, n)

    # Plot CalFire reference underneath
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

    # Draw late (large) perimeters first so early (small) ones sit on top
    for idx in reversed(range(n)):
        row = gofer_gdf.iloc[idx]
        frac = min(idx / t95_idx, 1.0)
        color = cmap(frac)
        gpd.GeoDataFrame([row], crs="EPSG:4326").plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            facecolor=color,
            edgecolor=color,
            alpha=0.01,
            linewidth=1.0,
            zorder=4 + (n - idx),
        )

    _add_colorbar(fig, ax, cmap)
    ax.set_title(title)

    if calfire_gdf is not None:
        ax.legend(loc='upper right')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.close()


def plot_perimeter_comparison(
    gofer_gdf: gpd.GeoDataFrame,
    ds: xr.Dataset,
    calfire_gdf: gpd.GeoDataFrame,
    title: str = "GOFER vs FRAP — Final Perimeter Comparison",
    save_path: str = None,
):
    """
    Compare the final fire perimeter from GOFER against the FRAP/CalFire
    reference perimeter on a streetmap basemap.

    Only the final (last timestep) GOFER perimeter is plotted, alongside
    the CalFire reference.

    Args:
        gofer_gdf: GeoDataFrame from raster_to_polygon. If multi-timestep,
            only the last timestep is used.
        ds: The source xarray Dataset for extent.
        calfire_gdf: CalFire/FRAP reference perimeter GeoDataFrame.
        title: Plot title.
        save_path: If provided, save the figure to this path.
    """
    extent = _extent_from_ds(ds)
    fig, ax = _setup_basemap(extent)

    # Get final GOFER perimeter
    if 'time' in gofer_gdf.columns and len(gofer_gdf) > 1:
        gofer_final = gpd.GeoDataFrame([gofer_gdf.iloc[-1]], crs=gofer_gdf.crs)
    else:
        gofer_final = gofer_gdf

    # Plot CalFire/FRAP reference (blue)
    calfire_gdf.to_crs(epsg=4326).plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        facecolor='none',
        edgecolor='blue',
        linewidth=2.5,
        label='FRAP (CalFire)',
        zorder=4,
    )

    # Plot GOFER final perimeter (red)
    gofer_final.to_crs(epsg=4326).plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        facecolor='none',
        edgecolor='red',
        linewidth=2.5,
        label='GOFER',
        zorder=5,
    )

    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=12)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.close()
