from __future__ import annotations
import xarray as xr
import numpy as np
from pathlib import Path
from tqdm.dask import TqdmCallback

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


def validate_xy_coords(goes_ds: xr.Dataset) -> None:
    """
    Validate that a GOES Dataset has ABI fixed-grid coordinates.

    If the dataset doesn't have coordinates, then we can't do anything!
    """
    if "x" not in goes_ds.coords or "y" not in goes_ds.coords:
        raise ValueError("GOES dataset must have `x` and `y` fixed-grid coordinates.")


def resolve_data_vars(
    goes_ds: xr.Dataset,
    data_var: str | None
) -> list[str]:
    """
    Resolve the GOES data variables to orthorectify.

    If ``data_vars`` is provided, those variables are selected. Otherwise, all
    variables with both ``y`` and ``x`` dimensions are selected.
    """
    data_vars = [data_var] if data_var else vars_with_xy_dims(goes_ds)

    if not data_vars:
        raise ValueError("No GOES data variables with both `y` and `x` dimensions found.")

    missing = [name for name in data_vars if name not in goes_ds.data_vars]
    if missing:
        raise ValueError(f"GOES data variable(s) not found: {', '.join(missing)}")

    non_spatial = [
        name
        for name in data_vars
        if not {"x", "y"}.issubset(goes_ds[name].dims)
    ]
    if non_spatial:
        raise ValueError(
            "GOES data variable(s) must have both `y` and `x` dimensions: "
            + ", ".join(non_spatial)
        )

    return data_vars


def eval_and_save_nc(
    ds: xr.Dataset,
    save_path: str,
    chunk_size: tuple | None = None,
    chunks: dict | None = None,
    data_var: str = 'MaskConfidence',
    desc: str = '...',
    verbose: bool = True
):
    '''
    Saves netcdf file. Also acts to realize the dask computation graph, 
    evaluating the lazy computations so they don't get deferred later. 
    Can be used as a checkpoint for each portion of the pipeline.

    This is preferable over .compute() since some Datasets may be too 
    large to fit into memory when performing the computations. Thus, 
    the result will need to live somewhere other than RAM; hence the 
    need to save it to disk, then loading it back to evaluate the graph.

    chunk_size -> outgoing; requires a tuple
    chunk -> loading; require a dictionary (maybe?)
    '''
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f'Saving to {save_path}...')

    encoding = {}
    for name, da in ds.variables.items():
        # clear encodings, only MaskConfidence encodings matter
        da.encoding.clear()
        enc = {}

        # for anything with a float, have the fill value be np.nan
        if np.issubdtype(da.dtype, np.floating):
            enc["_FillValue"] = np.nan

        # pass in a chunk size. Unfortunately, to_netcdf doesn't infer this
        if name == data_var:
            enc["dtype"] = "uint8"
            enc["scale_factor"] = np.float32(0.01)
            enc["add_offset"] = np.float32(0.0)
            enc["_FillValue"] = np.uint8(255)
            enc["chunksizes"] = chunk_size
            enc["zlib"] = False

        if enc:
            encoding[name] = enc

    # realize the computation graph
    with TqdmCallback(desc=f'Computing and saving {desc}' if verbose else None):
        # swap from netcdf4. has loud harmless errors when writing to a 
        #   netcdf file that doesn't already exist
        ds.to_netcdf(
            str(save_path),
            mode="w",
            engine="h5netcdf", 
            encoding=encoding
        )

    # load it back
    return xr.open_dataset(str(save_path), chunks=chunks)
