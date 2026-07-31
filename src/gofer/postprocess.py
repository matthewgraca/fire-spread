"""
Post-processing and quality control for GOFER fire perimeter products.
"""
import xarray as xr
import numpy as np


def trim_inactive_timesteps(
    filepath: str,
    data_var: str = 'MaskConfidence'
) -> tuple[int, int]:
    """
    Find the bounds of the active fire period by reading the file directly
    with h5py to avoid HDF5 chunk cache accumulation.

    Returns (first_fire, last_change) indices for slicing.

    Args:
        filepath: Path to the netCDF file to scan.
        data_var: Name of the data variable to check.

    Returns:
        Tuple of (first_fire_index, last_change_index) inclusive.
    """
    import h5py

    with h5py.File(filepath, 'r') as f:
        mc = f[data_var]
        n_times = mc.shape[0]

        # Raw int8 values: tolerance of 1 accounts for encoding/decoding
        # rounding errors (1 unit = 0.01 in decoded space)
        atol = 1

        # Trim leading: find first timestep with any fire pixels
        first_fire = 0
        for t in range(n_times):
            if np.any(mc[t][:] > atol):
                first_fire = t
                break

        # Trim trailing: walk backward until we find a frame that differs
        last_change = n_times - 1
        prev_slice = mc[n_times - 1][:]
        for t in range(n_times - 2, first_fire - 1, -1):
            curr_slice = mc[t][:]
            if not np.allclose(curr_slice, prev_slice, atol=atol):
                last_change = t + 1
                break
            prev_slice = curr_slice

    return first_fire, last_change


def round_to(
    ds: xr.Dataset,
    data_var: str = 'MaskConfidence',
    decimals: int = 2
) -> xr.Dataset:
    """
    Rounds the given data variable to the given decimals
    """
    return ds.assign(**{data_var : np.round(ds[data_var], decimals)})


def binarize(
    ds: xr.Dataset,
    data_var: str = 'MaskConfidence',
    threshold: float = 0.95
) -> xr.Dataset:
    """
    Everything below the threshold = 0, everything gte the threshold = 1.
    """
    return ds.assign(
        **{data_var : xr.where(
            ds[data_var] < threshold, np.float32(0), np.float32(1.0)
        )}
    )


def enforce_cummax(
    ds: xr.Dataset,
    data_var: str = 'MaskConfidence'
) -> xr.Dataset:
    """
    Enforces cumulative maximum along the time dimension.

    After compositing, early perimeter adjustment, smoothing, and binarization,
    the monotonicity of the perimeter product is not guaranteed. This function
    re-applies cummax to ensure each pixel is non-decreasing over time.
    """
    data = ds[data_var].values  # (time, lat, lon)
    np.maximum.accumulate(data, axis=0, out=data)
    return ds.assign(**{data_var: (ds[data_var].dims, data)})

                     
