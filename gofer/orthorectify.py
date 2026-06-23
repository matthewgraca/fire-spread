'''
Adapted from https://github.com/spestana/goes-ortho
'''
import annotationlib
from typing import Iterable
import numpy as np
import rioxarray
import xarray as xr


GRS80_ECCENTRICITY = 0.0818191910435


def _get_projection_params(goes_ds: xr.Dataset) -> dict[str, float]:
    """
    Extract GOES-R ABI geostationary projection parameters.

    GOES ABI fixed-grid files usually store these on the
    `goes_imager_projection` variable as CF projection attributes.
    """
    if "goes_imager_projection" not in goes_ds:
        raise ValueError("GOES dataset is missing `goes_imager_projection`.")

    proj = goes_ds["goes_imager_projection"]

    required = [
        "semi_major_axis",
        "semi_minor_axis",
        "perspective_point_height",
        "longitude_of_projection_origin",
    ]
    missing = [name for name in required if name not in proj.attrs]
    if missing:
        raise ValueError(
            "GOES projection variable is missing required attributes: "
            + ", ".join(missing)
        )

    req = float(proj.attrs["semi_major_axis"])
    rpol = float(proj.attrs["semi_minor_axis"])
    h = float(proj.attrs["perspective_point_height"]) + req
    lon_0 = float(proj.attrs["longitude_of_projection_origin"])

    return {
        "semi_major_axis": req,
        "semi_minor_axis": rpol,
        "satellite_height": h,
        "longitude_of_projection_origin": lon_0,
        "eccentricity": GRS80_ECCENTRICITY,
    }


def _open_dem_epsg4326(
    dem_filepath: str,
    bbox: Optional[tuple[float, float, float, float]] = None,
) -> xr.DataArray:
    """
    Open a SRTMv3 GeoTIFF DEM already in EPSG:4326.

    Returns a 2D DataArray with dims y, x and coordinates x=longitude,
    y=latitude. Nodata values are converted to NaN.
    """
    dem = rioxarray.open_rasterio(dem_filepath, masked=True)

    # SRTM GeoTIFF is normally one band. Drop band dimension.
    if "band" in dem.dims:
        dem = dem.squeeze("band", drop=True)

    if dem.rio.crs is not None and dem.rio.crs.to_epsg() != 4326:
        raise ValueError(
            f"DEM must be EPSG:4326, but its CRS is {dem.rio.crs!r}."
        )

    if bbox is not None:
        min_lon, min_lat, max_lon, max_lat = bbox

        dem = dem.rio.clip_box(
            minx=min_lon,
            miny=min_lat,
            maxx=max_lon,
            maxy=max_lat,
            crs="EPSG:4326",
        )

    # Keep terrain as float so NaNs are valid.
    dem = dem.astype("float64")

    # If masked=True did not pick up nodata, handle common metadata manually.
    nodata = dem.rio.nodata
    if nodata is not None and not np.isnan(nodata):
        dem = dem.where(dem != nodata)

    return dem


def _lonlat_to_abi_scan_angles(
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    elevation_m: np.ndarray,
    *,
    satellite_height: float,
    semi_major_axis: float,
    semi_minor_axis: float,
    eccentricity: float,
    longitude_of_projection_origin: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert lon/lat/elevation to GOES ABI fixed-grid scan angles.

    Inputs:
      lon_deg, lat_deg: degrees
      elevation_m: meters above ellipsoid
      satellite_height, semi_major_axis, semi_minor_axis: meters

    Returns:
      x, y scan angles in radians.
    """
    lon = np.radians(lon_deg)
    lat = np.radians(lat_deg)
    lon_0 = np.radians(longitude_of_projection_origin)

    # Geocentric latitude.
    lat_geo = np.arctan((semi_minor_axis**2 / semi_major_axis**2) * np.tan(lat))

    # Distance from Earth center to ellipsoid surface at geocentric latitude.
    rc = semi_minor_axis / np.sqrt(
        1.0 - eccentricity**2 * np.cos(lat_geo) ** 2
    )

    # Add terrain height. Treat NaN DEM cells as ellipsoid height.
    z = np.nan_to_num(elevation_m, nan=0.0)
    rc = rc + z

    sx = satellite_height - rc * np.cos(lat_geo) * np.cos(lon - lon_0)
    sy = -rc * np.cos(lat_geo) * np.sin(lon - lon_0)
    sz = rc * np.sin(lat_geo)

    y = np.arctan(sz / sx)
    x = np.arcsin(-sy / np.sqrt(sx**2 + sy**2 + sz**2))

    return x, y


def _make_ortho_map(
    goes_ds: xr.Dataset,
    dem_filepath: str,
    bbox: Optional[tuple[float, float, float, float]] = None
) -> xr.Dataset:
    """
    Build a DEM-grid Dataset containing:
      - longitude
      - latitude
      - elevation
      - dem_px_angle_x
      - dem_px_angle_y
    """
    params = _get_projection_params(goes_ds)
    dem = _open_dem_epsg4326(dem_filepath, bbox)

    lon_2d, lat_2d = np.meshgrid(dem["x"].values, dem["y"].values)
    elevation = dem.values

    abi_x, abi_y = _lonlat_to_abi_scan_angles(
        lon_2d,
        lat_2d,
        elevation,
        satellite_height=params["satellite_height"],
        semi_major_axis=params["semi_major_axis"],
        semi_minor_axis=params["semi_minor_axis"],
        eccentricity=params["eccentricity"],
        longitude_of_projection_origin=params["longitude_of_projection_origin"],
    )

    ortho_map = xr.Dataset(
        data_vars={
            "elevation": (("latitude", "longitude"), elevation),
        },
        coords={
            "longitude": ("longitude", dem["x"].values),
            "latitude": ("latitude", dem["y"].values),
            "dem_px_angle_x": (("latitude", "longitude"), abi_x),
            "dem_px_angle_y": (("latitude", "longitude"), abi_y),
        },
        attrs={
            "orthorectification": "GOES ABI sampled onto EPSG:4326 DEM grid",
            "dem_file": dem_filepath,
            "dem_crs": "EPSG:4326",
            "longitude_of_projection_origin": params[
                "longitude_of_projection_origin"
            ],
            "semi_major_axis": params["semi_major_axis"],
            "semi_minor_axis": params["semi_minor_axis"],
            "satellite_height": params["satellite_height"],
            "grs80_eccentricity": params["eccentricity"],
        },
    )

    return ortho_map


def _vars_with_xy_dims(goes_ds: xr.Dataset) -> list[str]:
    """
    Return GOES variables that can be spatially sampled on x/y.

    This intentionally excludes scalar metadata variables and projection vars.
    """
    out = []
    for name, da in goes_ds.data_vars.items():
        dims = set(da.dims)
        if {"x", "y"}.issubset(dims):
            out.append(name)
    return out


def _nearest_fixed_grid_coords(
    goes_ds: xr.Dataset,
    ortho_map: xr.Dataset,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each DEM cell, find the actual nearest GOES fixed-grid x/y coordinate.
    Useful for diagnostics and for zone labels.
    """
    flat_x = ortho_map["dem_px_angle_x"].values.ravel()
    flat_y = ortho_map["dem_px_angle_y"].values.ravel()

    nearest_x = goes_ds.sel(x=flat_x, method="nearest")["x"].values
    nearest_y = goes_ds.sel(y=flat_y, method="nearest")["y"].values

    shape = ortho_map["dem_px_angle_x"].shape
    return nearest_x.reshape(shape), nearest_y.reshape(shape)


def _zone_labels(fixed_x: np.ndarray, fixed_y: np.ndarray) -> np.ndarray:
    """
    Assign an integer label to each unique sampled GOES x/y pixel footprint.
    """
    pairs = np.column_stack([fixed_x.ravel(), fixed_y.ravel()])
    _, inverse = np.unique(pairs, axis=0, return_inverse=True)
    return inverse.reshape(fixed_x.shape)


def orthorectify(
    goes_filepath: str,
    dem_filepath: str,
    bbox: Optional[tuple[float, float, float, float]] = None
) -> xr.Dataset:
    """
    Orthorectify a GOES ABI netCDF file to a SRTMv3 DEM grid.

    Parameters
    ----------
    goes_filepath:
        String path to a GOES netCDF file. The file must contain x/y fixed-grid
        coordinates and `goes_imager_projection`.

    dem_filepath:
        String path to a SRTMv3 GeoTIFF file. The DEM is assumed to already be
        EPSG:4326.

    bbox:
        Optional bounding box in EPSG:4326 coordinates:

            (min_lon, min_lat, max_lon, max_lat)

        If provided, the DEM is clipped to this bounding box before
        orthorectification. 

    Returns
    -------
    xr.Dataset
        Orthorectified GOES dataset on DEM latitude/longitude coordinates.
        All GOES data variables with both `y` and `x` dimensions are sampled
        onto the DEM grid using nearest-neighbor lookup in ABI scan-angle space.
    """
    goes_ds = xr.open_dataset(goes_filepath, decode_times=False)

    if "x" not in goes_ds.coords or "y" not in goes_ds.coords:
        raise ValueError("GOES dataset must have `x` and `y` fixed-grid coordinates.")

    ortho = _make_ortho_map(goes_ds, dem_filepath, bbox)

    data_vars = _vars_with_xy_dims(goes_ds)
    if not data_vars:
        raise ValueError("No GOES data variables with both `y` and `x` dimensions found.")

    # Build 2D DataArray indexers. xarray will sample each GOES variable at the
    # nearest ABI fixed-grid coordinate corresponding to each DEM cell.
    x_indexer = ortho["dem_px_angle_x"]
    y_indexer = ortho["dem_px_angle_y"]

    sampled = goes_ds[data_vars].sel(x=x_indexer, y=y_indexer, method="nearest")

    # After vectorized indexing, sampled arrays use latitude/longitude dims
    # inherited from the 2D indexers. Merge them into the ortho output.
    for name in data_vars:
        ortho[name] = sampled[name]

    fixed_x, fixed_y = _nearest_fixed_grid_coords(goes_ds, ortho)
    ortho["abi_fixed_grid_x"] = (("latitude", "longitude"), fixed_x)
    ortho["abi_fixed_grid_y"] = (("latitude", "longitude"), fixed_y)
    ortho["zone_labels"] = (
        ("latitude", "longitude"),
        _zone_labels(fixed_x, fixed_y),
    )

    # Preserve useful non-spatial coordinates/metadata when present.
    for coord_name in goes_ds.coords:
        if coord_name not in {"x", "y"} and coord_name not in ortho.coords:
            coord = goes_ds[coord_name]
            if coord.ndim == 0:
                ortho = ortho.assign_coords({coord_name: coord})

    ortho.attrs.update(
        {
            "source_goes_file": goes_filepath,
            "source_dem_file": dem_filepath,
            "orthorectified_variables": ", ".join(data_vars),
            "resampling": "nearest",
        }
    )

    return ortho


### NOTE comparison, move later?
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def get_goes_cartopy_crs(goes_ds):
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


def add_goes_xy_meters(goes_ds):
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
def plot_original_vs_ortho(
    goes_filepath,
    ortho_ds,
    variable="Rad",
    extent=None,
    cmap="gray",
):
    goes_ds = xr.open_dataset(goes_filepath, decode_times=False)
    goes_ds = add_goes_xy_meters(goes_ds)
    goes_crs = get_goes_cartopy_crs(goes_ds)

    if extent is None:
        extent = [
            float(ortho_ds.longitude.min()),
            float(ortho_ds.longitude.max()),
            float(ortho_ds.latitude.min()),
            float(ortho_ds.latitude.max()),
        ]

    vmin, vmax = np.nanpercentile(
        ortho_ds[variable].values,
        [2, 98],
    )

    fig = plt.figure(figsize=(15, 6))

    ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
    im1 = goes_ds[variable].plot.pcolormesh(
        ax=ax1,
        x="x_m",
        y="y_m",
        transform=goes_crs,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        add_colorbar=False,
    )
    ax1.set_extent(extent, crs=ccrs.PlateCarree())
    ax1.coastlines(resolution="10m", linewidth=0.8)
    ax1.set_title("Original GOES fixed-grid")

    ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
    im2 = ortho_ds[variable].plot.pcolormesh(
        ax=ax2,
        x="longitude",
        y="latitude",
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        add_colorbar=False,
    )
    ax2.set_extent(extent, crs=ccrs.PlateCarree())
    ax2.coastlines(resolution="10m", linewidth=0.8)
    ax2.set_title("Orthorectified GOES")

    cbar = fig.colorbar(
        im2,
        ax=[ax1, ax2],
        orientation="vertical",
        shrink=0.85,
        pad=0.03,
    )
    cbar.set_label(variable)

    return fig, (ax1, ax2)

### NOTE GIF
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import imageio.v2 as imageio
from PIL import Image


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

    plt.tight_layout()
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)


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
    goes_ds = add_goes_xy_meters(goes_ds)
    goes_crs = get_goes_cartopy_crs(goes_ds)

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

    original_png = "_goes_original_frame.png"
    terrain_png = "_goes_terrain_corrected_frame.png"

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

    return output_gif
