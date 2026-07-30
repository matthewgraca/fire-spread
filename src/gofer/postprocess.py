"""
Post-processing and quality control for GOFER fire perimeter products.
"""
import xarray as xr
import numpy as np


def trim_inactive_timesteps(
    ds: xr.Dataset,
    data_var: str = 'MaskConfidence'
) -> xr.Dataset:
    """
    Trim leading timesteps with no fire detected and trailing timesteps
    where the perimeter has stopped growing.

    Since the product is a cumulative max, trailing frames that are identical
    to their predecessor contain no new information.

    Args:
        ds: Dataset with a time dimension and a binary fire variable.
        data_var: Name of the data variable to check.

    Returns:
        Dataset trimmed to the active fire period.
    """
    n_times = ds.sizes['time']

    # Trim leading: find first timestep with any fire pixels
    first_fire = 0
    for t in range(n_times):
        slice_t = ds[data_var].isel(time=t).values
        if slice_t.any():
            first_fire = t
            break

    # Trim trailing: walk backward until we find a frame that differs
    last_change = n_times - 1
    prev_slice = ds[data_var].isel(time=n_times - 1).values
    for t in range(n_times - 2, first_fire - 1, -1):
        curr_slice = ds[data_var].isel(time=t).values
        if not np.allclose(curr_slice, prev_slice, atol=0.01):
            last_change = t + 1
            break
        prev_slice = curr_slice

    return ds.isel(time=slice(first_fire, last_change + 1))


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

                     
