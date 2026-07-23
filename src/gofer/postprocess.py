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
    da = ds[data_var]

    # Trim leading: find first timestep with any fire pixels
    has_fire = da.any(dim=['latitude', 'longitude'])
    first_fire = int(has_fire.argmax(dim='time'))

    # Trim trailing: walk backward until we find a frame that differs
    # from the one before it
    last_change = len(da.time) - 1
    for t in range(len(da.time) - 1, 0, -1):
        diff = (da.isel(time=t) != da.isel(time=t - 1)).any().compute()
        if diff:
            last_change = t
            break

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

                     
