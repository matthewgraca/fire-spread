"""
Active fire confidence visualization.

Produces an animated GIF showing hourly active fire detections overlaid on
the cumulative fire perimeter. Each frame shows:
- A street-map basemap (CartoDB Voyager, matching fire_perimeter style)
- The burned perimeter up to that hour (dark fill for spatial context)
- The active fire confidence for that hour (hot colormap overlay)
- A timestamp annotation

Since ActiveFireConfidence is instantaneous (not cumulative), a static
image cannot capture its temporal dynamics — hence the animated GIF.
Also provides a single-frame snapshot function for reports.
"""
import io
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

from viz.gofer.tilers import CartoDBTiles

import warnings
warnings.filterwarnings("ignore", message="Cartopy created the following directory to cache")


# --- Colormap for active fire intensity ---

_ACTIVE_FIRE_COLORS = [
    (0.00, "#440154"),  # deep purple (low confidence)
    (0.15, "#482878"),
    (0.30, "#B73779"),
    (0.45, "#F0605D"),
    (0.60, "#FDAE61"),
    (0.75, "#FCFFA4"),
    (1.00, "#FFFFFF"),  # white-hot (high confidence)
]


def _active_fire_cmap():
    cmap = LinearSegmentedColormap.from_list("active_fire", _ACTIVE_FIRE_COLORS, N=256)
    cmap.set_under("none")  # below vmin → fully transparent
    return cmap


def _extent_from_ds(ds: xr.Dataset, buffer: float = 0.05) -> list:
    return [
        float(ds.longitude.min()) - buffer,
        float(ds.longitude.max()) + buffer,
        float(ds.latitude.min()) - buffer,
        float(ds.latitude.max()) + buffer,
    ]


def _setup_basemap(extent: list, figsize: tuple = (16, 12), zoom: int = 12):
    tiler = CartoDBTiles(style='rastertiles/voyager', cache=True)
    fig, ax = plt.subplots(
        1, 1, figsize=figsize,
        subplot_kw={'projection': ccrs.PlateCarree()},
        layout='constrained',
    )
    ax.add_image(tiler, zoom)
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    return fig, ax


def _add_colorbar(fig, ax, cmap, norm, threshold: float):
    """Add a persistent colorbar for active fire confidence."""
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', shrink=0.7, pad=0.03)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(['0.00', '0.25', '0.50', '0.75', '1.00'])
    label = "Fire Detection Confidence"
    if threshold > 0:
        label += f" (threshold > {threshold:.2f})"
    cbar.set_label(label)
    return cbar


def plot_active_fire_gif(
    ds: xr.Dataset,
    save_path: str,
    perimeter_var: str = "MaskConfidence",
    active_fire_var: str = "ActiveFireConfidence",
    fps: int = 6,
    step: int = 1,
    confidence_threshold: float = 0.0,
    dpi: int = 120,
    figsize: tuple = (16, 12),
    zoom: int = 12,
):
    """
    Create an animated GIF of active fire confidence over time.

    Each frame shows a basemap with the cumulative perimeter in dark fill
    and the instantaneous active fire confidence as a heatmap overlay.

    Args:
        ds: Dataset containing both perimeter and active fire variables.
        save_path: Output path for the GIF file.
        perimeter_var: Name of the binary perimeter variable.
        active_fire_var: Name of the active fire confidence variable.
        fps: Frames per second in the output GIF.
        step: Render every Nth timestep (1 = all, 3 = every 3rd hour).
        confidence_threshold: Only plot pixels with confidence strictly
            above this value. Examples:
              0.0  → plot all non-zero detections (default)
              0.5  → plot only > 50% confidence
              0.95 → plot only > 95% confidence (highest intensity)
        dpi: Resolution of each frame.
        figsize: Figure size in inches (width, height).
        zoom: Basemap tile zoom level (higher = sharper, slower). Default 12.
    """
    perimeter = ds[perimeter_var].values  # (T, H, W)
    active_fire = ds[active_fire_var].values  # (T, H, W)
    times = ds["time"].values
    n_times = perimeter.shape[0]

    lat = ds["latitude"].values
    lon = ds["longitude"].values
    extent = _extent_from_ds(ds)

    # Norm always spans 0–1 for consistent colorbar across frames
    cmap = _active_fire_cmap()
    norm = mcolors.Normalize(vmin=max(confidence_threshold, 1e-6), vmax=1.0)

    # Subsample timesteps
    indices = list(range(0, n_times, step))

    frames = []
    for t in indices:
        fig, ax = _setup_basemap(extent, figsize=figsize, zoom=zoom)

        # Perimeter background: dark fill
        perim_frame = np.ma.masked_where(perimeter[t] == 0, perimeter[t])
        ax.pcolormesh(
            lon, lat, perim_frame,
            transform=ccrs.PlateCarree(),
            cmap="Greys",
            vmin=0, vmax=1,
            alpha=0.5,
            zorder=3,
        )

        # Active fire overlay — apply user threshold
        af_frame = active_fire[t].copy()
        af_frame[af_frame <= confidence_threshold] = np.nan
        af_masked = np.ma.masked_invalid(af_frame)

        if af_masked.count() > 0:
            ax.pcolormesh(
                lon, lat, af_masked,
                transform=ccrs.PlateCarree(),
                cmap=cmap,
                norm=norm,
                alpha=0.9,
                zorder=4,
            )

        # Colorbar
        _add_colorbar(fig, ax, cmap, norm, confidence_threshold)

        # Timestamp title
        time_str = pd.Timestamp(times[t]).strftime("%Y-%m-%d %H:%M")
        ax.set_title(f"Active Fire — {time_str}", fontsize=14, fontweight="bold")

        # Render to PIL image
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        buf.close()
        plt.close(fig)

    # Save as GIF
    if frames:
        duration_ms = int(1000 / fps)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(
            save_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
        )


def plot_active_fire_snapshot(
    ds: xr.Dataset,
    time_idx: int,
    save_path: str | None = None,
    perimeter_var: str = "MaskConfidence",
    active_fire_var: str = "ActiveFireConfidence",
    confidence_threshold: float = 0.0,
    figsize: tuple = (16, 12),
    zoom: int = 12,
):
    """
    Plot a single-frame snapshot of active fire confidence at a given timestep.

    Shows a basemap with the cumulative perimeter and active fire confidence
    overlay. Useful for inspecting a specific hour or for inclusion in reports.

    Args:
        ds: Dataset containing both perimeter and active fire variables.
        time_idx: Index along the time dimension to plot.
        save_path: If provided, save the figure to this path.
        perimeter_var: Name of the binary perimeter variable.
        active_fire_var: Name of the active fire confidence variable.
        confidence_threshold: Only plot pixels with confidence strictly
            above this value. Default 0.0 (all non-zero values shown).
        figsize: Figure size in inches.
        zoom: Basemap tile zoom level (higher = sharper, slower). Default 12.

    Returns:
        (fig, ax) if save_path is None, for further customization.
    """
    perimeter = ds[perimeter_var].isel(time=time_idx).values
    active_fire = ds[active_fire_var].isel(time=time_idx).values
    time_val = ds["time"].values[time_idx]

    lat = ds["latitude"].values
    lon = ds["longitude"].values
    extent = _extent_from_ds(ds)

    cmap = _active_fire_cmap()
    norm = mcolors.Normalize(vmin=max(confidence_threshold, 1e-6), vmax=1.0)

    fig, ax = _setup_basemap(extent, figsize=figsize, zoom=zoom)

    # Perimeter background
    perim_masked = np.ma.masked_where(perimeter == 0, perimeter)
    ax.pcolormesh(
        lon, lat, perim_masked,
        transform=ccrs.PlateCarree(),
        cmap="Greys",
        vmin=0, vmax=1,
        alpha=0.5,
        zorder=3,
    )

    # Active fire overlay
    af_frame = active_fire.copy()
    af_frame[af_frame <= confidence_threshold] = np.nan
    af_masked = np.ma.masked_invalid(af_frame)

    if af_masked.count() > 0:
        ax.pcolormesh(
            lon, lat, af_masked,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            norm=norm,
            alpha=0.9,
            zorder=4,
        )

    _add_colorbar(fig, ax, cmap, norm, confidence_threshold)

    time_str = pd.Timestamp(time_val).strftime("%Y-%m-%d %H:%M")
    ax.set_title(f"Active Fire — {time_str}", fontsize=14, fontweight="bold")

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        return fig, ax
