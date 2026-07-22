"""
Converts a binary fire perimeter raster into simplified polygon(s).
"""
import numpy as np
import xarray as xr
import geopandas as gpd
import rasterio.transform
from shapely.geometry import shape
from rasterio.features import shapes

from gofer.spatial_smoothing import estimate_pixel_size_m


def _get_transform(lon: np.ndarray, lat: np.ndarray):
    """
    Build a rasterio affine transform from lon/lat coordinate arrays.

    Expects lat in descending order (top-left origin).
    """
    dx = float(lon[1] - lon[0])
    dy = float(lat[1] - lat[0])  # negative for descending lat

    return rasterio.transform.from_bounds(
        lon[0] - dx / 2,
        lat[-1] + dy / 2,
        lon[-1] + dx / 2,
        lat[0] - dy / 2,
        width=len(lon),
        height=len(lat),
    )


def _vectorize_single_frame(
    mask: np.ndarray,
    transform,
) -> gpd.GeoDataFrame:
    """
    Vectorize a single 2D binary mask into polygon(s).
    """
    fire_shapes = [
        (shape(geom), value)
        for geom, value in shapes(mask, mask=(mask == 1), transform=transform)
        if value == 1
    ]

    if not fire_shapes:
        return None

    geometries = [geom for geom, _ in fire_shapes]
    gdf = gpd.GeoDataFrame(geometry=geometries, crs="EPSG:4326")
    return gdf.union_all()


def _get_tolerance_in_degrees(ds: xr.Dataset, simplify_factor: float) -> float:
    # Compute simplification tolerance in degrees
    pixel_height_m, pixel_width_m = estimate_pixel_size_m(ds)
    nominal_pixel_size_m = np.sqrt(pixel_height_m * pixel_width_m)
    tolerance_m = nominal_pixel_size_m * simplify_factor

    mean_lat = float(np.nanmean(ds["latitude"].values))
    meters_per_degree = 111_320 * np.cos(np.deg2rad(mean_lat))
    tolerance_deg = tolerance_m / meters_per_degree

    return tolerance_deg


def raster_to_polygon(
    ds: xr.Dataset,
    data_var: str = "MaskConfidence",
    simplify_factor: float = 2.0,
) -> gpd.GeoDataFrame:
    """
    Convert a binary raster Dataset into simplified fire perimeter polygon(s).

    The raster is expected to contain a binary variable (1 = fire, 0 = no fire).
    The spatial resolution is derived from the coordinate spacing of the Dataset,
    and polygons are simplified with a maximum error margin of
    simplify_factor * pixel_size (following the GOFER 2:1 design ratio).

    If the Dataset contains a time dimension, a polygon is produced for each
    timestep.

    Args:
        ds: xarray Dataset with latitude/longitude coordinates and a binary
            data variable.
        data_var: Name of the binary variable to vectorize.
        simplify_factor: Multiplier on pixel size for polygon simplification
            tolerance. Default 2.0 matches the original GOFER 2:1 ratio.

    Returns:
        GeoDataFrame containing fire perimeter polygon(s) in EPSG:4326.
        If the input has a time dimension, each row corresponds to a timestep
        with a 'time' column.
    """
    da = ds[data_var]
    has_time = "time" in da.dims

    # Coordinate arrays
    lon = ds["longitude"].values
    lat = ds["latitude"].values

    # rasterio expects top-left origin (lat descending)
    lat_flipped = lat[0] < lat[-1]
    if lat_flipped:
        lat = lat[::-1]

    transform = _get_transform(lon, lat)
    tolerance_deg = _get_tolerance_in_degrees(ds, simplify_factor)

    # Vectorize each frame
    records = []

    if has_time:
        times = ds["time"].values
        for i, t in enumerate(times):
            mask = da.isel(time=i).values.astype(np.uint8)
            if lat_flipped:
                mask = np.flipud(mask)

            polygon = _vectorize_single_frame(mask, transform)
            if polygon is not None:
                simplified = polygon.simplify(tolerance_deg)
                records.append({"time": t, "geometry": simplified})
    else:
        mask = da.values.astype(np.uint8)
        if lat_flipped:
            mask = np.flipud(mask)

        polygon = _vectorize_single_frame(mask, transform)
        if polygon is not None:
            simplified = polygon.simplify(tolerance_deg)
            records.append({"geometry": simplified})

    if not records:
        cols = ["time", "geometry"] if has_time else ["geometry"]
        return gpd.GeoDataFrame(columns=cols, crs="EPSG:4326")

    return gpd.GeoDataFrame(records, crs="EPSG:4326")
