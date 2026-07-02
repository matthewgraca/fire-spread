from __future__ import annotations

import xarray as xr
import numpy as np

def _dynamic_chunk(
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

    West and East may differ in time. Therefore, you must pass in a sequence 
    of hourly dates. The times of the Datasets will be normalized to the 
    nearest hour, then aligned and imputed according to those dates.

    Any empty gaps in timestep (due to corrupted timestep or satellite gaps) 
    will be filled with zeros.

    Arguments:
        west_ds: The GOES-West Dataset.
        east_ds: The GOES-East Dataset.
        dates: The sequence of dates the two datasets will be aligned with
        data_var: The data variable subject to concatentation.

    Returns:
        xr.Dataset: A dataset merged according to the minimum data variable.
        Also spatially chunks the data for memory efficiency.
    """
    # normalize time
    west_ds = west_ds.assign_coords(time=west_ds.time.dt.round("1h"))
    east_ds = east_ds.assign_coords(time=east_ds.time.dt.round("1h"))

    # prune invalid timesteps due to improper storage on goes's end
    target_time = xr.DataArray(dates, dims="time", name="time")

    west_valid_dates = west_ds.time.isin(dates)
    east_valid_dates = east_ds.time.isin(dates)

    west_ds = west_ds.sel(time=west_valid_dates)
    east_ds = east_ds.sel(time=east_valid_dates)

    # reindex to target time
    west_ds = west_ds.reindex(
        time=target_time,
        fill_value={data_var: np.float32(0)}
    )
    east_ds = east_ds.reindex(
        time=target_time,
        fill_value={data_var: np.float32(0)}
    )

    # concat
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

    # merge on minimum value
    combined_ds = combined_ds.mean(dim="satellite", skipna=True)

    return _dynamic_chunk(combined_ds, data_var, target_chunk_mb=256)
