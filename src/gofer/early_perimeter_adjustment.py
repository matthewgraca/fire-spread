from typing import Any

import xarray as xr
import numpy as np
from gofer.spatial_smoothing import smooth
from gofer.ortho import *
from tqdm import tqdm


def get_scaling_factors(
    ds: xr.Dataset,
    ortho_kwargs: dict[str, Any],
    data_var: str = 'MaskConfidence',
    show_progress: bool = True
) -> np.ndarray:
    '''
    For each hour h:

        Gather the confidences from hour [1, h]
        Calculate the cumulative max of the confidences from [1, h]
        Parallax adjustment
        Apply spatial smoothing
        Calculate the max confidence of the entire frame s_h; the scaling factor
            (+ exceptions like t >= 500)
        For the entire frame, divide each pixel by s_h

    We perform our own cumulative max since built-in methods take a ton of 
    auxiliary space.
    '''
    MAX_TIME = 500
    times = ds['time'].values
    capped_times = times[:MAX_TIME]
    scaling_factors = [1.0] * len(times)
    running = None
    ortho_map = make_ortho_map(ds, **ortho_kwargs)
    if show_progress:
        tqdm.write(
            f'Calculating scaling factors for the first '
            f'{len(capped_times)} timesteps'
        )
    for i, t in (
        enumerate(tqdm(capped_times))
        if show_progress else enumerate(capped_times)
    ):
        ds_t = ds.isel({'time' : i}).load()
        ds_t = apply_ortho_map(ds_t, ortho_map, data_var=data_var)
        ds_t = smooth(ds_t, input_variable=data_var)

        if running is None:
            running = ds_t[data_var].values
        else:
            running = xr.apply_ufunc(np.fmax, running, ds_t[data_var], dask='allowed')

        s_t = float(np.max(running))
        '''
        tqdm.write(
            f'{t} '
            f'min: {ds_t[data_var].min().values}, '
            f'max: {ds_t[data_var].max().values}, '
            f'mean: {ds_t[data_var].mean().values}, '
        )
        tqdm.write(
            f'{t} '
            f'min: {np.min(running).item()}, '
            f'max: {np.max(running).item()}, '
            f'mean: {np.mean(running).item()}, \n'
        )
        '''
        
        scaling_factors[i] = s_t

    #tqdm.write(f'Intermediary scaling factors: {scaling_factors}')

    # exceptions: 
    #   no scalers < 0.1
    #   0.0 gets set to 1.0 to avoid division by 0
    sf = np.array(scaling_factors)
    sf = np.where(sf < 0.1, 1.0, sf)

    #tqdm.write(f'After cleanup scaling factors: {sf.tolist()}')

    return sf

def apply_scaling_factors(
    ds: xr.Dataset,
    scaling_factors: np.ndarray,
    data_var: str = 'MaskConfidence'
) -> xr.Dataset:
    early_perimeter_scales_da = xr.DataArray(
        scaling_factors,
        dims=('time',),
        coords={'time': ds['time']},
    ).astype(ds[data_var].dtype)
    return ds.assign(
        **{data_var : ds[data_var] / early_perimeter_scales_da}
    )
