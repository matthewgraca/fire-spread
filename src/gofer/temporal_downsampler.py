import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from gofer.remapper import map_fdc_mask_to_confidence
from gofer.goes_utils import eval_and_save_nc


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


def _prune_invalid_and_pad_missing_timesteps(
    ds: xr.Dataset,
    dates: pd.DatetimeIndex,
    data_var: str = 'MaskConfidence'
) -> xr.Dataset:
    '''
    Due to improper storage on GOES's end, there may be some invalid timesteps.
    We prune those improper timesteps, and fill out the data of any timesteps
    that are missing with NaN (to be resolved downstream by the caller).
    '''
    # prune invalid timesteps due to improper storage on goes's end
    target_time = xr.DataArray(dates.tz_localize(None), dims="time", name="time")
    valid_dates = ds.time.isin(target_time)
    ds = (ds
        .sel(time=valid_dates)
        .reindex(
            time=target_time,
            fill_value={data_var: np.nan}
        )
    )

    return ds

def _downsample(ds: xr.Dataset, hour: pd.Timestamp) -> xr.Dataset:
    ''' 
    Given a dataset with multiple timesteps, all variables will be merged to a 
    given hour by their max observation/value.

    e.g.
        Given 3:00 as the hour,
        times: [2:05, 2:10, 3:02] -> 3:00
        values: [1, 5, 2] -> 5
    '''
    return (ds
        .max(dim='time')
        .expand_dims(time=[pd.Timestamp(hour, tz='UTC').tz_localize(None)])
    )

def _cummax(
    ds: xr.Dataset,
    data_var: str = 'MaskConfidence',
    running_cummax: np.ndarray | None = None
) -> xr.Dataset:
    '''
    Performs a max on the given data and running cumulative max 
    array.

    Essentially just compares the max between the running max and the 
    current dataset, returning a dataset with the max between the two, 
    and the running max.
    '''
    ds = ds.load() # get eager
    curr_data = ds[data_var].data
    running = (
        curr_data if running_cummax is None 
        else np.fmax(running_cummax, curr_data)
    )
    cummax_da = xr.DataArray(
        running,
        dims=("time", "y", "x"),
        coords={
            "time": ds["time"],
            "y": ds["y"],
            "x": ds["x"],
        },
        name=data_var,
        attrs=ds[data_var].attrs,
    )
    cummax_ds = ds.assign(**{data_var : cummax_da})

    return cummax_ds, running

def _clean_ds(
    ds: xr.Dataset,
    keep_coords: dict[str] = {'time', 'y', 'x'},
    keep_vars: dict[str] = {'MaskConfidence', 'goes_imager_projection'},
    keep_attrs: dict[str] = {
        'orbital_slot', 'platform_ID', 'dataset_name', 
        'active_fire', 'fire_perimeter', 'fire_name'
    }
) -> xr.Dataset:
    """
    Removes a ton of coords and attributes we'll no longer need.
    """
    ds_clean = ds.copy()
    remove_coords = [
        coord for coord in ds_clean.coords
        if coord not in keep_coords
    ]
    ds_clean = ds_clean.drop_vars(remove_coords)

    remove_data_vars = [
        data_var for data_var in ds_clean.data_vars
        if data_var not in keep_vars
    ]
    ds_clean = ds_clean.drop_vars(remove_data_vars)

    remove_attrs = [
        attr for attr in ds_clean.attrs
        if attr not in keep_attrs
    ]
    for attr in remove_attrs:
       ds_clean.attrs.pop(attr, None)

    return ds_clean 


def aggregate(
    goes_save_dir: str,
    csv_path: str,
    temp_dir: str,
    dates: pd.DatetimeIndex,
    data_var: str = 'MaskConfidence',
    fire_name: str = 'N/A',
    is_perimeter: bool = True
) -> xr.Dataset:
    '''
    Pipeline:
        1. Remamp -- Maps FDC Mask data to confidence values
        2. Temporal downsample -- gathers all subhourly observations and 
            groups them into hourly observations
        3. Imputation -- any gaps in the data are filled to ensure a 
            temporally resolved dataset

    To prevent maxing out on RAM, each frame is saved into an nc file.

    Args:
        goes_save_dir (str): The directory pointing to the location of the 
            saved goes data. Will be appended with the files found in csv_path 
            to generate a complete file path of the GOES netcdf file.
        temp_dir (str): The directory that will contain the intermediate 
            hourly nc files that will get merged into one dataset later.
        csv_path (str): The path of the csv file that contains an inventory 
            of the files that were ingested.
        dates (pd.DatetimeIndex): A DatetimeIndex with which to align the 
            dates in the dataset with.
        data_var (str): The data variable to extract and aggregate on.
        fire_name (str): Name of the fire.
        is_perimeter (bool): Determines if the frames will be active fire 
            (frames stay as-is), or fire perimeter (cumulative max of frames).

    Returns:
        xr.Dataset: A dataset with the temporally downsampled dataset.
    '''
    files_df = _read_csv(csv_path)
    dataset_paths = []
    out_dir = Path(temp_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    running_cummax = None
    for hour, hour_df in (pbar := tqdm(files_df.groupby('timestamp'))):
        pbar.set_description(f'Processing {hour}')

        ds = _open_and_combine_ds(
            goes_save_dir=goes_save_dir,
            goes_filepaths=hour_df['file'].to_list()
        )

        ds = map_fdc_mask_to_confidence(ds)

        ds = _downsample(ds, hour)

        if is_perimeter:
            ds, running_cummax = _cummax(ds, data_var, running_cummax)
            ds = ds.assign_attrs(
                fire_name=fire_name,
                perimeter="True",
                description="Perimeter product, containing the cumulative max "
                "of the confidences of the past active fire pixels"
            )
        else:
            ds = ds.assign_attrs(
                fire_name=fire_name,
                active_fire="True",
                description="Active fire product, containing the  "
                "the confidence of the current active fire pixels"
            )

        ds = _clean_ds(ds)

        path = out_dir / Path(hour.isoformat() + '.nc')
        ds = eval_and_save_nc(ds, path, data_var=data_var, verbose=False) 
        ds.close()

        dataset_paths.append(path)

    ds = xr.open_mfdataset(
        dataset_paths,
        combine='nested',
        concat_dim='time',
        chunks={'time' : 1}
    )

    ds = _prune_invalid_and_pad_missing_timesteps(ds, dates, data_var)

    # ffill gaps for fire perimeter (to respect cummax)
    # impute with 0 for active fire
    if is_perimeter:
        ds[data_var] = ds[data_var].ffill(dim='time')
    else:
        ds[data_var] = ds[data_var].fillna(0)

    ds = ds.chunk({'time' : 1})

    return ds
