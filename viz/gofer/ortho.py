import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import imageio.v2 as imageio
from PIL import Image
from pathlib import Path

def _get_goes_cartopy_crs(goes_ds):
    proj = goes_ds["goes_imager_projection"]

    semi_major_axis = float(proj.attrs["semi_major_axis"])
    semi_minor_axis = float(proj.attrs["semi_minor_axis"])
    perspective_point_height = float(proj.attrs["perspective_point_height"])
    lon_origin = float(proj.attrs["longitude_of_projection_origin"])
    sweep_axis = proj.attrs.get("sweep_angle_axis", "x")

    return ccrs.Geostationary(
        central_longitude=lon_origin,
        satellite_height=perspective_point_height,
        globe=ccrs.Globe(
            semimajor_axis=semi_major_axis,
            semiminor_axis=semi_minor_axis,
        ),
        sweep_axis=sweep_axis,
    )


def _add_goes_xy_meters(goes_ds):
    """
    Add Cartopy-compatible projected x/y coordinates in meters.

    GOES ABI x/y coordinates are scan angles in radians.
    Cartopy Geostationary expects projection coordinates in meters.
    """
    h = float(
        goes_ds["goes_imager_projection"].attrs["perspective_point_height"]
    )

    return goes_ds.assign_coords(
        x_m=("x", goes_ds["x"].values * h),
        y_m=("y", goes_ds["y"].values * h),
    )


def _fig_to_rgb_array(fig):
    """
    Convert a Matplotlib figure to an RGB numpy array.
    """
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    return buf.reshape(h, w, 3)


def _save_frame(
    da,
    *,
    label,
    output_png,
    x,
    y,
    transform,
    extent,
    cmap,
    vmin,
    vmax,
    title=None,
):
    """
    Save one map frame as a PNG.
    """
    fig = plt.figure(figsize=(8, 7), dpi=140)
    ax = plt.axes(projection=ccrs.PlateCarree())

    da.plot.pcolormesh(
        ax=ax,
        x=x,
        y=y,
        transform=transform,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        add_colorbar=False,
    )

    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.coastlines(resolution="10m", linewidth=0.8)
    ax.add_feature(cfeature.STATES, linewidth=0.4)

    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.3,
        alpha=0.5,
        linestyle="--",
    )
    gl.top_labels = False
    gl.right_labels = False

    if title is not None:
        ax.set_title(title)

    # Large label on image
    ax.text(
        0.01,
        0.1,
        label,
        transform=ax.transAxes,
        fontsize=18,
        fontweight="bold",
        va="top",
        ha="left",
        bbox={
            "facecolor": "white",
            "alpha": 0.75,
            "edgecolor": "none",
            "boxstyle": "round,pad=0.3",
        },
    )

    plt.tight_layout()
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)

# NOTE grab the folders before the file to save it in the desired location
def make_original_vs_terrain_corrected_gif(
    goes_filepath,
    ortho_ds,
    variable="Rad",
    output_gif="goes_terrain_correction_comparison.gif",
    extent=None,
    cmap="gray",
    duration_ms=800,
):
    """
    Create a GIF cycling between original GOES fixed-grid data and
    terrain-corrected orthorectified GOES data.

    Parameters
    ----------
    goes_filepath : str
        Path to the original GOES netCDF file.

    ortho_ds : xr.Dataset
        Dataset returned by orthorectify().

    variable : str
        Variable to plot, for example "Rad" or "CMI".

    output_gif : str
        Output GIF filename.

    extent : list or tuple, optional
        Cartopy extent in lon/lat order:

            [min_lon, max_lon, min_lat, max_lat]

        If omitted, uses the orthorectified dataset bounds.

    cmap : str
        Matplotlib colormap.

    duration_ms : int
        Duration of each GIF frame in milliseconds.

    Returns
    -------
    str
        Path to the saved GIF.
    """
    goes_ds = xr.open_dataset(goes_filepath, decode_times=False)
    goes_ds = _add_goes_xy_meters(goes_ds)
    goes_crs = _get_goes_cartopy_crs(goes_ds)

    if extent is None:
        extent = [
            float(ortho_ds.longitude.min()),
            float(ortho_ds.longitude.max()),
            float(ortho_ds.latitude.min()),
            float(ortho_ds.latitude.max()),
        ]

    # Use the orthorectified subset for shared local contrast.
    plot_values = ortho_ds[variable].where(np.isfinite(ortho_ds[variable]))
    vmin, vmax = np.nanpercentile(plot_values.values, [2, 98])

    output_dir = Path(output_gif).parent
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    original_png = str(output_dir / Path("_goes_original_frame.png"))
    terrain_png = str(output_dir / Path("_goes_terrain_corrected_frame.png"))

    print('Creating frames (this will take a bit of time...)')

    _save_frame(
        goes_ds[variable],
        label="Original",
        output_png=original_png,
        x="x_m",
        y="y_m",
        transform=goes_crs,
        extent=extent,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        title="Original GOES Fixed Grid",
    )

    _save_frame(
        ortho_ds[variable],
        label="Terrain Corrected",
        output_png=terrain_png,
        x="longitude",
        y="latitude",
        transform=ccrs.PlateCarree(),
        extent=extent,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        title="Orthorectified GOES",
    )

    frames = [
        Image.open(original_png).convert("RGB"),
        Image.open(terrain_png).convert("RGB"),
    ]

    frames[0].save(
        output_gif,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )

    print(
        f'Complete! Saved to \n'
        f'\t{original_png}\n'
        f'\t{terrain_png}\n'
        f'\t{output_gif}'
    )

    return output_gif
