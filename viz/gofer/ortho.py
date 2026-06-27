import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path
from gofer.goes_utils import *
from viz.gofer.goes_plotting_utils import *
from gofer.geometry import lonlat_to_abi_scan_angles

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

#### NOTE This portion compares Datasets; the first dataset is in the native GOES projection, and the second is in lat/lon.
# The methods above just read a file and a Dataset; will eventually be merged in a clean-up pass.

def _default_extent_from_latlon_ds(ds: xr.Dataset) -> list[float]:
    """
    Return Cartopy extent from an orthorectified lat/lon Dataset.

    Cartopy extent order:
        [min_lon, max_lon, min_lat, max_lat]
    """
    if "longitude" not in ds.coords or "latitude" not in ds.coords:
        raise ValueError(
            "terrain_corrected_ds must contain `longitude` and `latitude` "
            "coordinates if extent is not provided."
        )

    return [
        float(ds["longitude"].min()),
        float(ds["longitude"].max()),
        float(ds["latitude"].min()),
        float(ds["latitude"].max()),
    ]


def _shared_color_limits(
    original_da: xr.DataArray,
    terrain_da: xr.DataArray,
    *,
    percentiles: tuple[float, float] = (2, 98),
    force_range: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """
    Compute shared color limits for both frames.
    """
    if force_range is not None:
        return float(force_range[0]), float(force_range[1])

    values = []

    for da in [original_da, terrain_da]:
        arr = da.values
        arr = arr[np.isfinite(arr)]
        if arr.size:
            values.append(arr.ravel())

    if not values:
        raise ValueError("No finite values found for color scaling.")

    all_values = np.concatenate(values)
    vmin, vmax = np.nanpercentile(all_values, percentiles)

    return float(vmin), float(vmax)


def _decorate_map(
    ax,
    *,
    extent: list[float],
    label: str,
    title: str | None,
    show_features: bool = True,
    show_gridlines: bool = True,
):
    """
    Add common map decorations and frame label.
    """
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

    if title is not None:
        ax.set_title(title)

    ax.text(
        0.03,
        0.95,
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


def _save_native_abi_frame(
    original_ds: xr.Dataset,
    *,
    variable: str,
    output_png: str | Path,
    extent: list[float],
    label: str,
    title: str | None,
    cmap: str,
    vmin: float,
    vmax: float,
    show_features: bool,
    show_gridlines: bool,
):
    """
    Save a frame from a native GOES ABI fixed-grid Dataset.
    """
    ds_plot = add_goes_xy_meters(original_ds)
    goes_crs = get_goes_cartopy_crs(ds_plot)

    fig = plt.figure(figsize=(8, 7), dpi=140)
    ax = plt.axes(projection=ccrs.PlateCarree())

    ds_plot[variable].plot.pcolormesh(
        ax=ax,
        x="x_m",
        y="y_m",
        transform=goes_crs,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        add_colorbar=False,
        infer_intervals=False,
    )

    _decorate_map(
        ax,
        extent=extent,
        label=label,
        title=title,
        show_features=show_features,
        show_gridlines=show_gridlines,
    )

    fig.savefig(output_png, dpi=140)
    plt.close(fig)


def _save_terrain_corrected_frame(
    terrain_corrected_ds: xr.Dataset,
    *,
    variable: str,
    output_png: str | Path,
    extent: list[float],
    label: str,
    title: str | None,
    cmap: str,
    vmin: float,
    vmax: float,
    show_features: bool,
    show_gridlines: bool,
):
    """
    Save a frame from an orthorectified latitude/longitude Dataset.
    """
    if "longitude" not in terrain_corrected_ds.coords:
        raise ValueError("terrain_corrected_ds is missing `longitude` coordinate.")

    if "latitude" not in terrain_corrected_ds.coords:
        raise ValueError("terrain_corrected_ds is missing `latitude` coordinate.")

    fig = plt.figure(figsize=(8, 7), dpi=140)
    ax = plt.axes(projection=ccrs.PlateCarree())

    terrain_corrected_ds[variable].plot.pcolormesh(
        ax=ax,
        x="longitude",
        y="latitude",
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        add_colorbar=False,
        infer_intervals=False,
    )

    _decorate_map(
        ax,
        extent=extent,
        label=label,
        title=title,
        show_features=show_features,
        show_gridlines=show_gridlines,
    )

    fig.savefig(output_png, dpi=140)
    plt.close(fig)


def make_original_vs_terrain_corrected_gif2(
    original_ds: xr.Dataset,
    terrain_corrected_ds: xr.Dataset,
    *,
    variable: str = "MaskConfidence",
    output_gif: str | Path = "original_vs_terrain_corrected.gif",
    extent: list[float] | None = None,
    original_label: str = "Original",
    terrain_corrected_label: str = "Terrain Corrected",
    original_title: str = "Original GOES ABI fixed grid",
    terrain_corrected_title: str = "Terrain-corrected GOES",
    cmap: str = "gray",
    duration_ms: int = 900,
    percentiles: tuple[float, float] = (2, 98),
    force_range: tuple[float, float] | None = None,
    show_features: bool = True,
    show_gridlines: bool = True,
    keep_frames: bool = False,
) -> str:
    """
    Create a GIF cycling between a native ABI Dataset and an orthorectified Dataset.

    Parameters
    ----------
    original_ds:
        Dataset on the native GOES ABI fixed grid. It must contain:
            - variable
            - x/y scan-angle coordinates
            - goes_imager_projection

    terrain_corrected_ds:
        Orthorectified Dataset on a latitude/longitude grid. It must contain:
            - variable
            - longitude/latitude coordinates

    variable:
        Variable name to plot from both datasets.

    output_gif:
        Output GIF path.

    extent:
        Optional Cartopy extent in lon/lat order:

            [min_lon, max_lon, min_lat, max_lat]

        If omitted, the extent is inferred from terrain_corrected_ds.

    force_range:
        Optional fixed color range.

        For MaskConfidence, use:

            force_range=(0.0, 1.0)

    Returns
    -------
    str
        Path to saved GIF.
    """
    if variable not in original_ds:
        raise ValueError(f"original_ds does not contain variable `{variable}`.")

    if variable not in terrain_corrected_ds:
        raise ValueError(
            f"terrain_corrected_ds does not contain variable `{variable}`."
        )

    if extent is None:
        extent = _default_extent_from_latlon_ds(terrain_corrected_ds)

    output_gif = Path(output_gif)
    output_gif.parent.mkdir(parents=True, exist_ok=True)

    vmin, vmax = _shared_color_limits(
        original_ds[variable],
        terrain_corrected_ds[variable],
        percentiles=percentiles,
        force_range=force_range,
    )

    original_png = output_gif.with_name(f"{output_gif.stem}_original_frame.png")
    terrain_png = output_gif.with_name(
        f"{output_gif.stem}_terrain_corrected_frame.png"
    )

    _save_native_abi_frame(
        original_ds,
        variable=variable,
        output_png=original_png,
        extent=extent,
        label=original_label,
        title=original_title,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        show_features=show_features,
        show_gridlines=show_gridlines,
    )

    _save_terrain_corrected_frame(
        terrain_corrected_ds,
        variable=variable,
        output_png=terrain_png,
        extent=extent,
        label=terrain_corrected_label,
        title=terrain_corrected_title,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        show_features=show_features,
        show_gridlines=show_gridlines,
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

    for frame in frames:
        frame.close()

    if not keep_frames:
        original_png.unlink(missing_ok=True)
        terrain_png.unlink(missing_ok=True)

    return str(output_gif)
