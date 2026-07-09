from __future__ import annotations

import xarray as xr
import numpy as np

def dynamic_chunk(
    ds: xr.Dataset,
    data_var: str = "MaskConfidence",
    target_chunk_mb: float = 128,
) -> xr.Dataset:
    """
    Dynamically chunks the Dataset to a given target chunk size.

    Originally, we chunked data according to time = 1, lat = n, lon = m.

    However, this produced chunks that were smaller than the recommended range 
    (10MB, 1GB). Since Dask scheduling has overhead of 1ms, it's worth having 
    larger chunks if we're going to do it at all.

    So, we now chunk time dynamically.
    """
    bytes_per_value = ds[data_var].dtype.itemsize

    lat_chunk_size = ds.sizes["latitude"]
    lon_chunk_size = ds.sizes["longitude"]

    bytes_per_time_slice = lat_chunk_size * lon_chunk_size * bytes_per_value
    target_bytes = target_chunk_mb * 1024**2

    available_time_chunks = max(1, int(target_bytes // bytes_per_time_slice))
    time_chunk_size = min(available_time_chunks, ds.sizes["time"])

    return ds.chunk({
        "time": time_chunk_size,
        "latitude": lat_chunk_size,
        "longitude": lon_chunk_size,
    })


def composite(
    west_ds: xr.Dataset,
    east_ds: xr.Dataset,
    dates: pd.DatetimeIndex,
    data_var: str
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

    Returns:
        xr.Dataset: A dataset merged according to the minimum data variable.
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

    return dynamic_chunk(combined_ds, data_var, target_chunk_mb=256)
