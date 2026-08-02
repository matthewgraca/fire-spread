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
    filepaths = [Path(goes_save_dir) / Path(f) for f in goes_filepaths]

    # Extract t (seconds since 2000-01-01 12:00:00) from each file manually.
    # t is a scalar coordinate that xarray does not reliably concatenate.
    origin = pd.Timestamp("2000-01-01 12:00:00")
    t_values = []
    for fp in filepaths:
        with xr.open_dataset(fp, decode_times=False) as tmp:
            t_val = tmp["t"].values
            if np.ndim(t_val) == 0:
                t_values.append(float(t_val))
            else:
                t_values.extend(t_val.astype(float).tolist())
    decoded_times = origin + pd.to_timedelta(t_values, unit="s")

    ds = xr.open_mfdataset(
        filepaths,
        concat_dim="time",
        combine="nested",
        data_vars="all",
        coords="minimal",
        compat="override",
        drop_variables=drop_variables + ["t"],
        decode_times=False,
        parallel=False,
    )

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

    # Build fill_value for all data variables
    fill_values = {
        name: np.nan for name in ds.data_vars
        if np.issubdtype(ds[name].dtype, np.floating)
    }

    ds = (ds
        .sel(time=valid_dates)
        .reindex(
            time=target_time,
            fill_value=fill_values
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
    keep_vars: dict[str] = {'MaskConfidence', 'ActiveFireConfidence', 'goes_imager_projection'},
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


def _process_hour(goes_save_dir: str, goes_filepaths: list[str], hour, out_dir: str) -> str:
    """Process a single hour: open, remap, downsample, clean, save."""
    from gofer.goes_utils import MC_ENCODING

    ds = _open_and_combine_ds(
        goes_save_dir=goes_save_dir,
        goes_filepaths=goes_filepaths
    )
    ds = map_fdc_mask_to_confidence(ds)
    ds = _downsample(ds, hour)
    ds = _clean_ds(ds)
    path = Path(out_dir) / Path(hour.isoformat() + '.nc')
    encoding = {name: {'dtype': 'float32'} for name in ds.coords if ds.coords[name].dtype.kind == 'f'}
    encoding['MaskConfidence'] = MC_ENCODING
    ds.to_netcdf(str(path), mode="w", engine="scipy", encoding=encoding)
    ds.close()
    return str(path)


def aggregate(
    goes_save_dir: str,
    csv_path: str,
    temp_dir: str,
    dates: pd.DatetimeIndex,
    data_var: str = 'MaskConfidence',
    fire_name: str = 'N/A',
    is_perimeter: bool = True,
    verbose: bool = True,
    max_workers: int = 12
) -> xr.Dataset:
    '''
    Pipeline:
        1. Remap -- Maps FDC Mask data to confidence values
        2. Temporal downsample -- gathers all subhourly observations and 
            groups them into hourly observations (parallelized)
        3. Cumulative max -- applies running cummax for perimeter mode
        4. Imputation -- any gaps in the data are filled to ensure a 
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
        max_workers (int): Number of threads for parallel downsampling.

    Returns:
        xr.Dataset: A dataset with the temporally downsampled dataset.
    '''
    from concurrent.futures import ProcessPoolExecutor, as_completed

    files_df = _read_csv(csv_path)
    out_dir = Path(temp_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hourly_groups = list(files_df.groupby('timestamp'))

    # Phase 1: Parallel downsample (open, remap, downsample, clean, save)
    dataset_paths = []
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _process_hour, goes_save_dir, hour_df['file'].to_list(), hour, str(out_dir)
            ): hour
            for hour, hour_df in hourly_groups
        }
        with tqdm(total=len(futures), disable=not verbose, leave=False,
                  delay=1, desc="Downsampling") as pbar:
            for future in as_completed(futures):
                dataset_paths.append(future.result())
                pbar.update(1)

    dataset_paths.sort()

    # Phase 2: Sequential cummax (if perimeter mode)
    # ActiveFireConfidence is created here from the pre-cummax hourly values.
    from gofer.goes_utils import MC_ENCODING, AFC_ENCODING
    _encoding = {'MaskConfidence': MC_ENCODING, 'ActiveFireConfidence': AFC_ENCODING}

    if is_perimeter:
        running_cummax = None
        for path in tqdm(dataset_paths, disable=not verbose, leave=False,
                         delay=1, desc="Applying cummax"):
            ds = xr.open_dataset(str(path)).load()
            # Preserve the pre-cummax value as ActiveFireConfidence
            ds['ActiveFireConfidence'] = ds[data_var].copy()
            ds, running_cummax = _cummax(ds, data_var, running_cummax)
            ds = ds.assign_attrs(
                fire_name=fire_name,
                perimeter="True",
                description="Perimeter product, containing the cumulative max "
                "of the confidences of the past active fire pixels"
            )
            ds.to_netcdf(str(path), mode="w", engine="scipy", encoding=_encoding)
            ds.close()
    else:
        _encoding_af = {'MaskConfidence': MC_ENCODING}
        # Just tag attributes for active fire mode
        for path in dataset_paths:
            ds = xr.open_dataset(str(path)).load()
            ds = ds.assign_attrs(
                fire_name=fire_name,
                active_fire="True",
                description="Active fire product, containing the "
                "the confidence of the current active fire pixels"
            )
            ds.to_netcdf(str(path), mode="w", engine="scipy", encoding=_encoding_af)
            ds.close()

    # Phase 3: Combine and impute
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
        # ActiveFireConfidence is instantaneous: fill gaps with 0
        if 'ActiveFireConfidence' in ds.data_vars:
            ds['ActiveFireConfidence'] = ds['ActiveFireConfidence'].fillna(0)
    else:
        ds[data_var] = ds[data_var].fillna(0)

    ds = ds.chunk({'time' : 1})

    return ds
