from gofer.remapper import fdc_mask_confidence_dataset
from viz.gofer.goes_plotting_utils import (
    get_goes_cartopy_crs, add_goes_xy_meters
)
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def plot_mask_confidence_native(
    ds: xr.Dataset,
    *,
    extent=None,
    cmap="inferno",
    title="GOES FDC Mask Confidence",
    show_features=True,
    show_gridlines=True,
    add_colorbar=True,
):
    """
    Plot MaskConfidence from a non-orthorectified GOES FDC Dataset.

    This assumes the dataset is still on the native GOES ABI fixed grid
    with coordinates `x` and `y` in scan-angle radians.

    Parameters
    ----------
    ds:
        Dataset containing `MaskConfidence`, `x`, `y`, and
        `goes_imager_projection`.

    extent:
        Optional map extent in lon/lat order:

            [min_lon, max_lon, min_lat, max_lat]

        Example:

            [-120.25, -118.75, 37.25, 38.50]

    cmap:
        Matplotlib colormap.

    title:
        Plot title.

    show_features:
        If True, draw coastlines, states, and borders.

    show_gridlines:
        If True, draw labeled gridlines.

    add_colorbar:
        If True, add a colorbar.

    Returns
    -------
    fig, ax, im
    """
    if "MaskConfidence" not in ds:
        raise ValueError("Dataset does not contain `MaskConfidence`.")

    if "goes_imager_projection" not in ds:
        raise ValueError("Dataset is missing `goes_imager_projection`.")

    if "x" not in ds.coords or "y" not in ds.coords:
        raise ValueError("Dataset must contain native GOES `x` and `y` coordinates.")

    ds_plot = add_goes_xy_meters(ds)
    goes_crs = get_goes_cartopy_crs(ds_plot)

    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    im = ds_plot["MaskConfidence"].plot.pcolormesh(
        ax=ax,
        x="x_m",
        y="y_m",
        transform=goes_crs,
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        add_colorbar=add_colorbar,
        cbar_kwargs={"label": "Mask confidence"} if add_colorbar else None,
        infer_intervals=False,
    )

    if extent is not None:
        ax.set_extent(extent, crs=ccrs.PlateCarree())

    if show_features:
        ax.coastlines(resolution="10m", linewidth=0.8)
        ax.add_feature(cfeature.STATES, linewidth=0.4)
        ax.add_feature(cfeature.BORDERS, linewidth=0.4)

    if show_gridlines:
        gl = ax.gridlines(
            draw_labels=True,
            linewidth=0.3,
            alpha=0.5,
            linestyle="--",
        )
        gl.top_labels = False
        gl.right_labels = False

    ax.set_title(title)

    return fig, ax, im


goes_filepath = '/home/mgraca/Workspace/fire-spread/data/goes/noaa-goes17/ABI-L2-FDCC/2020/255/08/OR_ABI-L2-FDCC-M6_G17_s20202550801176_e20202550803549_c20202550804107.nc'
before = xr.open_dataset(goes_filepath, decode_times=False)
print(before)

ds = fdc_mask_confidence_dataset(goes_filepath)
print(ds)
print()
print(ds['MaskConfidence'].values.max())
print(ds['MaskConfidence'].values.min())
print(ds['MaskConfidence'].values.mean())
# bobcat
# (-118.10447665373637, -117.76645404123737, 34.16559282525563, 34.48392452230033)
fig, ax, im = plot_mask_confidence_native(
    ds,
    extent=(-118.10447665373637, -117.76645404123737, 34.16559282525563, 34.48392452230033),
    title="Native GOES FDC Mask Confidence",
)
plt.show()
