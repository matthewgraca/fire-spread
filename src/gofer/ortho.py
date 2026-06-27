from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import rioxarray
import xarray as xr

from gofer.geometry import lonlat_to_abi_scan_angles
from gofer.goes_utils import (
    get_projection_params, resolve_data_vars, validate_xy_coords, 
)


BBox = tuple[float, float, float, float]


def _open_dem_epsg4326(
    dem_filepath: str,
    bbox: Optional[BBox] = None,
) -> xr.DataArray:
    """
    Open a DEM GeoTIFF in EPSG:4326.

    Returns a 2D DataArray with dimensions ``y`` and ``x`` and coordinates
    ``x=longitude`` and ``y=latitude``. Nodata values are converted to NaN.
    """
    dem = rioxarray.open_rasterio(dem_filepath, masked=True)

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

    dem = dem.astype("float64")

    nodata = dem.rio.nodata
    if nodata is not None and not np.isnan(nodata):
        dem = dem.where(dem != nodata)

    return dem


def _make_ortho_map(
    goes_ds: xr.Dataset,
    dem_filepath: str,
    bbox: BBox,
    *,
    include_fixed_grid_diagnostics: bool = True,
) -> xr.Dataset:
    """
    Build reusable orthorectification geometry on the DEM grid.

    The returned Dataset contains latitude/longitude coordinates, DEM elevation,
    ABI scan-angle indexers, and optionally nearest fixed-grid diagnostics. This
    geometry can be reused for every frame in a multi-time GOES Dataset as long
    as the fixed grid and projection are unchanged.

    Fixed grid diagnostics refer to which original ABI pixels the DEM-grid 
    sampled from. Recall that DEM is sub-pixel (30m) from the perspective 
    of GOES (2km); so we have to make a choice on what pixel on the DEM-grid 
    will take from GOES, where we use nearest-neighbor.
    
    abi_fixed_grid_x, abi_fixed_grid_y, and zone_labels records the GOES grid 
    cells selected.

    This tells you things like how many of the same GOES pixel was used for 
    a given bundle of pixels on the DEM-grid.
    """
    validate_xy_coords(goes_ds)
    params = get_projection_params(goes_ds)
    dem_da = _open_dem_epsg4326(dem_filepath, bbox)

    lon_2d, lat_2d = np.meshgrid(dem_da["x"].values, dem_da["y"].values)
    elevation = dem_da.values

    abi_x, abi_y = lonlat_to_abi_scan_angles(
        lon_2d,
        lat_2d,
        elevation,
        satellite_height=params["satellite_height"],
        semi_major_axis=params["semi_major_axis"],
        semi_minor_axis=params["semi_minor_axis"],
        eccentricity=params["eccentricity"],
        longitude_of_projection_origin=params["longitude_of_projection_origin"],
    )

    attrs = {
        "orthorectification": "GOES ABI sampled onto EPSG:4326 DEM grid",
        "dem_crs": "EPSG:4326",
        "longitude_of_projection_origin": params["longitude_of_projection_origin"],
        "semi_major_axis": params["semi_major_axis"],
        "semi_minor_axis": params["semi_minor_axis"],
        "satellite_height": params["satellite_height"],
        "grs80_eccentricity": params["eccentricity"],
        "dem_file" : dem_filepath,
        "bbox" : str(bbox)
    }

    ortho_map = xr.Dataset(
        data_vars={
            "elevation": (("latitude", "longitude"), elevation),
        },
        coords={
            "longitude": ("longitude", dem_da["x"].values),
            "latitude": ("latitude", dem_da["y"].values),
            "dem_px_angle_x": (("latitude", "longitude"), abi_x),
            "dem_px_angle_y": (("latitude", "longitude"), abi_y),
        },
        attrs=attrs,
    )

    if include_fixed_grid_diagnostics:
        fixed_x, fixed_y = _nearest_fixed_grid_coords(goes_ds, ortho_map)
        ortho_map["abi_fixed_grid_x"] = (("latitude", "longitude"), fixed_x)
        ortho_map["abi_fixed_grid_y"] = (("latitude", "longitude"), fixed_y)
        ortho_map["zone_labels"] = (
            ("latitude", "longitude"),
            _zone_labels(fixed_x, fixed_y),
        )

    return ortho_map


def _nearest_fixed_grid_coords(
    goes_ds: xr.Dataset,
    ortho_map: xr.Dataset,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each DEM cell, find the nearest GOES fixed-grid x/y coordinate.

    These arrays are useful for diagnostics and for zone labels.
    """
    flat_x = ortho_map["dem_px_angle_x"].values.ravel()
    flat_y = ortho_map["dem_px_angle_y"].values.ravel()

    nearest_x = goes_ds.sel(x=flat_x, method="nearest")["x"].values
    nearest_y = goes_ds.sel(y=flat_y, method="nearest")["y"].values

    shape = ortho_map["dem_px_angle_x"].shape
    return nearest_x.reshape(shape), nearest_y.reshape(shape)


def _zone_labels(fixed_x: np.ndarray, fixed_y: np.ndarray) -> np.ndarray:
    """Assign an integer label to each unique sampled GOES x/y pixel footprint."""
    pairs = np.column_stack([fixed_x.ravel(), fixed_y.ravel()])
    _, inverse = np.unique(pairs, axis=0, return_inverse=True)
    return inverse.reshape(fixed_x.shape)


def _apply_ortho_map(
    goes_ds: xr.Dataset,
    ortho_map: xr.Dataset,
    *,
    data_vars: Optional[Sequence[str]] = None
) -> xr.Dataset:
    """
    Apply precomputed orthorectification geometry to a GOES Dataset.

    Non-spatial dimensions such as ``time`` are preserved. The fixed-grid
    dimensions ``y`` and ``x`` are replaced by ``latitude`` and ``longitude`` via
    vectorized nearest-neighbor indexing in ABI scan-angle space.
    """
    resolved_data_vars = resolve_data_vars(goes_ds, data_vars)

    x_indexer = ortho_map["dem_px_angle_x"]
    y_indexer = ortho_map["dem_px_angle_y"]

    sampled = goes_ds[resolved_data_vars].sel(
        x=x_indexer,
        y=y_indexer,
        method="nearest",
    )

    ortho = ortho_map.copy()
    for name in resolved_data_vars:
        ortho[name] = sampled[name]

    # keep non-spatial coords
    output_dims = set(ortho.dims)
    for coord_name, coord in goes_ds.coords.items():
        if coord_name in {"x", "y"} or coord_name in ortho.coords:
            continue
        if coord.ndim == 0 or set(coord.dims).issubset(output_dims):
            ortho = ortho.assign_coords({coord_name: coord})

    ortho.attrs.update(goes_ds.attrs)
    ortho.attrs.update(ortho_map.attrs)
    ortho.attrs.update(
        {
            "orthorectified_variables": ", ".join(resolved_data_vars),
            "resampling": "nearest",
            "geometry_reuse": (
                "DEM-to-ABI scan angles, nearest fixed-grid coordinates, and "
                "zone labels were computed once and reused for the input Dataset."
            ),
        }
    )

    return ortho


def orthorectify(
    goes_ds: xr.Dataset,
    *,
    dem_filepath: str,
    bbox: BBox,
    data_vars: Sequence[str] | None = None,
    include_fixed_grid_diagnostics: bool = True,
) -> xr.Dataset:
    """
    Orthorectify a GOES Dataset onto an EPSG:4326 DEM grid.

    This is the high-level pipeline function. It computes geometry once, then
    applies that geometry to all requested variables and all time frames in the
    Dataset.
    """
    ortho_map = _make_ortho_map(
        goes_ds=goes_ds,
        dem_filepath=dem_filepath,
        bbox=bbox,
        include_fixed_grid_diagnostics=include_fixed_grid_diagnostics,
    )

    ortho = _apply_ortho_map(
        goes_ds=goes_ds,
        ortho_map=ortho_map,
        data_vars=data_vars,
    )

    ortho.attrs.update(
        {
            "source_dem_file": dem_filepath,
        }
    )

    return ortho
