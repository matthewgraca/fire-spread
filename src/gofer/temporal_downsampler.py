import xarray as xr
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from gofer.remapper import map_fdc_mask_to_confidence


def _read_csv(path: str) -> pd.DataFrame:
    files_df = pd.read_csv(path, parse_dates=['start', 'end', 'creation'])
    files_df['timestamp'] = files_df['creation'].dt.ceil("h")
    return files_df


def _open_and_combine_ds(
    goes_save_dir: None,
    goes_filepaths: list[str],
    drop_variables=["Area", "Temp", "Power"]
) -> xr.Dataset:
    ds = xr.open_mfdataset(
        [Path(goes_save_dir) / Path(f) for f in goes_filepaths],
        concat_dim="time",
        combine="nested",
        drop_variables=drop_variables,
        decode_times=False,
        parallel=True,
        chunks={"time": 1, "y": 1500, "x": 2500}
    )
    
    # decode_times=True sometimes plays out poorly, so we'll do it manually
    # ds.t.attrs['units'] = seconds since 2000-01-01 12:00:00
    origin = pd.Timestamp("2000-01-01 12:00:00")
    decoded_times = origin + pd.to_timedelta(ds["t"].values, unit="s")
    ds = ds.assign_coords(time=decoded_times)

    return ds 


def _prune_invalid_timesteps(ds: xr.Dataset, dates: pd.DatetimeIndex) -> xr.Dataset:
    # prune invalid timesteps due to improper storage on goes's end
    target_time = xr.DataArray(dates.tz_localize(None), dims="time", name="time")
    valid_dates = ds.time.isin(target_time)
    return ds.sel(time=valid_dates)

def aggregate(
    goes_save_dir: str,
    csv_path: str,
    dates: pd.DatetimeIndex
) -> xr.Dataset:
    '''
    Temporally downsample a dataset according to a given list of dates.

    Any empty gaps in timestep (due to corrupted timestep or satellite gaps) 
    will be filled with zeros.

    For example, given a DataFrame of subhourly dates, the method will 
    then combine these subhourly dates into the hour according to max 
    on time.

    Aggregates subhourly data into hourly resolution by downsampling 5-minute 
    GOES data into hourly data by max confidence.

    A bit of an overloaded function in terms of seperation of duties, no?

    Args:
        files_df (pd.DataFrame): A DataFrame of files, with a filename and 
            a timestamp column.
        dates (pd.DatetimeIndex): A DatetimeIndex with which to align the 
            dates in the dataset with.

    Returns:
        xr.Dataset: A dataset with the temporally downsampled dataset.
    '''
    files_df = _read_csv(csv_path)
    datasets = []
    for hour, hour_df in (pbar := tqdm(files_df.groupby('timestamp'))):
        pbar.set_description(f'Processing {hour}')

        ds = _open_and_combine_ds(
            goes_save_dir=goes_save_dir,
            goes_filepaths=hour_df['file'].to_list()
        )

        ds = map_fdc_mask_to_confidence(ds)

        # merge subhourly on max confidence, then reindex the time
        ds = (ds
            .max(dim='time')
            .expand_dims(time=[pd.Timestamp(hour, tz='UTC').tz_localize(None)])
        )

        datasets.append(ds)

    ds = xr.concat(datasets, dim='time')
    ds = _prune_invalid_timesteps(ds, dates)

    return ds

