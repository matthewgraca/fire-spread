from __future__ import annotations
from typing import Iterable
import numpy as np
import rioxarray
import xarray as xr
from gofer.goes_utils import get_projection_params, vars_with_xy_dims

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

    Note that the act of orthorectification creates a base lat/lon grid, 
    effectively performing coordinate transformation from ABI scan angles to 
    lat/lon inherently.

    Furthermore, given a GOES East and a GOES West file, if the exact same 
    DEM and bbox is used, the lat/lon grid will be the same. Thus, they will 
    share a common grid, co-registering the two satellites.
    """
    params = get_projection_params(goes_ds)
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

    data_vars = vars_with_xy_dims(goes_ds)
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
