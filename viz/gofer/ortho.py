import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path
from gofer.goes_utils import get_projection_params
from gofer.orthorectify import lonlat_to_abi_scan_angles

def make_uncorrected_on_ortho_grid(
    goes_filepath: str,
    ortho_ds: xr.Dataset,
    variable: str = "Rad",
) -> xr.DataArray:
    """
    Sample original GOES data onto the orthorectified lat/lon grid
    without terrain correction.

    This creates a fair 'Original' comparison frame on the same grid as
    the terrain-corrected output.
    """
    goes_ds = xr.open_dataset(goes_filepath, decode_times=False)

    params = get_projection_params(goes_ds)

    lon_2d, lat_2d = np.meshgrid(
        ortho_ds["longitude"].values,
        ortho_ds["latitude"].values,
    )

    zero_elevation = np.zeros_like(lon_2d, dtype="float64")

    flat_x, flat_y = lonlat_to_abi_scan_angles(
        lon_2d,
        lat_2d,
        zero_elevation,
        satellite_height=params["satellite_height"],
        semi_major_axis=params["semi_major_axis"],
        semi_minor_axis=params["semi_minor_axis"],
        eccentricity=params["eccentricity"],
        longitude_of_projection_origin=params[
            "longitude_of_projection_origin"
        ],
    )

    flat_map = xr.Dataset(
        coords={
            "longitude": ortho_ds["longitude"],
            "latitude": ortho_ds["latitude"],
            "flat_px_angle_x": (("latitude", "longitude"), flat_x),
            "flat_px_angle_y": (("latitude", "longitude"), flat_y),
        }
    )

    sampled = goes_ds[variable].sel(
        x=flat_map["flat_px_angle_x"],
        y=flat_map["flat_px_angle_y"],
        method="nearest",
    )

    valid = (
        (flat_map["flat_px_angle_x"] >= goes_ds["x"].min()) &
        (flat_map["flat_px_angle_x"] <= goes_ds["x"].max()) &
        (flat_map["flat_px_angle_y"] >= goes_ds["y"].min()) &
        (flat_map["flat_px_angle_y"] <= goes_ds["y"].max())
    )

    sampled = sampled.where(valid)

    sampled.name = f"{variable}_original_on_latlon_grid"
    sampled.attrs.update(goes_ds[variable].attrs)
    sampled.attrs["terrain_corrected"] = "false"
    sampled.attrs["description"] = (
        "Original GOES data sampled onto the orthorectified lat/lon grid "
        "using zero elevation."
    )

    goes_ds.close()

    return sampled

def _save_plain_latlon_frame(
    da,
    *,
    output_png,
    label,
    cmap="gray",
    vmin=None,
    vmax=None,
    title=None,
):
    """
    Save a lat/lon gridded DataArray as a plain Matplotlib image.

    No Cartopy, no Shapely, no GeoAxes.
    """
    lon = da["longitude"].values
    lat = da["latitude"].values
    data = da.values

    extent = [
        float(np.nanmin(lon)),
        float(np.nanmax(lon)),
        float(np.nanmin(lat)),
        float(np.nanmax(lat)),
    ]

    # Many rasters have latitude descending.
    origin = "upper" if lat[0] > lat[-1] else "lower"

    fig, ax = plt.subplots(figsize=(8, 7), dpi=140)

    im = ax.imshow(
        data,
        extent=extent,
        origin=origin,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        aspect="auto",
    )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    if title is not None:
        ax.set_title(title)

    ax.text(
        0.03,
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

    fig.savefig(output_png, bbox_inches='tight')
    plt.close(fig)

def make_original_vs_terrain_corrected_gif(
    goes_filepath: str,
    ortho_ds: xr.Dataset,
    variable: str = "Rad",
    output_gif: str = "goes_original_vs_terrain_corrected.gif",
    cmap: str = "gray",
    duration_ms: int = 900,
):
    """
    Create a GIF cycling between:
      1. Original GOES sampled to the lat/lon grid with zero elevation.
      2. Terrain-corrected GOES from orthorectify().

    This avoids Cartopy entirely, preventing Shapely/GEOS savefig errors.
    """
    original_da = make_uncorrected_on_ortho_grid(
        goes_filepath=goes_filepath,
        ortho_ds=ortho_ds,
        variable=variable,
    )

    terrain_da = ortho_ds[variable]

    # Shared color limits.
    both = np.concatenate(
        [
            original_da.values[np.isfinite(original_da.values)].ravel(),
            terrain_da.values[np.isfinite(terrain_da.values)].ravel(),
        ]
    )

    vmin, vmax = np.nanpercentile(both, [2, 98])

    output_dir = Path(output_gif).parent
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    original_png = str(output_dir / Path("_frame_original.png"))
    terrain_png = str(output_dir / Path("_frame_terrain_corrected.png"))

    _save_plain_latlon_frame(
        original_da,
        output_png=original_png,
        label="Original",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        title="Original GOES, sampled to lat/lon grid",
    )

    _save_plain_latlon_frame(
        terrain_da,
        output_png=terrain_png,
        label="Terrain Corrected",
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

    return output_gif
