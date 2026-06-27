# Utilities for primarily plotting GOES datasets as-is, no transformations

import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def get_goes_cartopy_crs(goes_ds: xr.Dataset) -> ccrs.Geostationary:
    """
    Build a Cartopy Geostationary CRS from a native GOES ABI Dataset.
    """
    if "goes_imager_projection" not in goes_ds:
        raise ValueError("Dataset is missing `goes_imager_projection`.")

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


def add_goes_xy_meters(goes_ds: xr.Dataset) -> xr.Dataset:
    """
    Add Cartopy-compatible projected x/y coordinates in meters.

    GOES ABI x/y coordinates are scan angles in radians. Cartopy's
    Geostationary projection expects projected coordinates in meters.
    """
    if "goes_imager_projection" not in goes_ds:
        raise ValueError("Dataset is missing `goes_imager_projection`.")

    if "x" not in goes_ds.coords or "y" not in goes_ds.coords:
        raise ValueError("Dataset must contain native GOES `x` and `y` coordinates.")

    h = float(goes_ds["goes_imager_projection"].attrs["perspective_point_height"])

    return goes_ds.assign_coords(
        x_m=("x", goes_ds["x"].values * h),
        y_m=("y", goes_ds["y"].values * h),
    )

