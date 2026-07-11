from __future__ import annotations

import xarray as xr
import numpy as np


def composite(
    west_ds: xr.Dataset,
    east_ds: xr.Dataset,
    dates: pd.DatetimeIndex,
    data_var: str,
    keep_attrs: dict = {'fire_name', 'description', 'active_fire', 'fire_perimeter'}
) -> xr.Dataset:
    """
    Concatenates the dataset of east and west observations by time, filling  
    any empty timesteps with zeros.

    Datasets are then merged by the minimum agreed value between the two 
    satellites on the given variable.

    Arguments:
        west_ds: The GOES-West Dataset.
        east_ds: The GOES-East Dataset.
        dates: The sequence of dates the two datasets will be aligned with
        data_var: The data variable subject to concatentation.
        keep_attrs: Attributes to keep. Composite is the last fusing step, 
            so this is where we shed a ton of attributes.

    Returns:
        xr.Dataset: A dataset merged according to the mean data variable.
        Also spatially chunks the data for memory efficiency.
    """
    combined_ds = xr.concat(
        [west_ds[data_var], east_ds[data_var]],
        dim=xr.IndexVariable("satellite", ["GOESWEST", "GOESEAST"])
    )
    combined_ds = xr.DataArray(
        data=combined_ds.data,
        dims=combined_ds.dims,
        coords={
            "time": combined_ds["time"],
            "satellite": combined_ds["satellite"],
            "latitude": combined_ds["latitude"],
            "longitude": combined_ds["longitude"],
        },
        name=data_var,
        attrs={
            "long_name": "Composite GOES West and East fire mask confidence"
        }
    ).to_dataset()

    # merge on mean value (for GOFER-Combined)
    combined_ds = combined_ds.mean(dim="satellite", skipna=True)

    # carry over attributes you want
    combined_ds = combined_ds.assign_attrs(
        pipeline='composited',
        **{
            k: west_ds.attrs[k]
            for k in west_ds.attrs if k in keep_attrs 
        }
    )

    return combined_ds
