from __future__ import annotations
import xarray as xr
import numpy as np

GRS80_ECCENTRICITY = 0.0818191910435

def get_projection_params(goes_ds: xr.Dataset) -> dict[str, float]:
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

def vars_with_xy_dims(goes_ds: xr.Dataset) -> list[str]:
    """
    Return GOES variables that can be spatially sampled on x/y.

    This intentionally excludes scalar metadata variables and projection vars.

    If the output is empty, this implies that you have a GOES dataset without 
    xy spatial coordinates.
    """
    out = []
    for name, da in goes_ds.data_vars.items():
        dims = set(da.dims)
        if {"x", "y"}.issubset(dims):
            out.append(name)
    return out
