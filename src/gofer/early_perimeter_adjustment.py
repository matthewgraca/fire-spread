from typing import Any

import xarray as xr
import numpy as np
from gofer.spatial_smoothing import smooth
from gofer.ortho import *
from gofer.composite import dynamic_chunk
from tqdm import tqdm


def get_scaling_factors(
    ds: xr.Dataset,
    ortho_kwargs: dict[str, Any],
    data_var: str = 'MaskConfidence',
    show_progress: bool = True
) -> xr.Dataset:
    '''
    For each hour h:

        Gather the confidences from hour [1, h]
        Calculate the cumulative max of the confidences from [1, h]
        Parallax adjustment
        Apply spatial smoothing
        Calculate the max confidence of the entire frame s_h; the scaling factor
            (+ exceptions like t >= 500, 0.1 -> too low)
        For the entire frame, divide each pixel by s_h

    We perform our own cumulative max since built-in methods take a ton of 
    auxiliary space.
    '''
    MAX_TIME = 500
    times = ds['time'].values[:MAX_TIME]
    scaling_factors = [1.0] * len(times)
    running = None
    ortho_map = make_ortho_map(ds, **ortho_kwargs)
    if show_progress:
        tqdm.write(
            f'Calculating scaling factors for the first '
            f'{MAX_TIME if len(times) >= MAX_TIME else len(times)} timesteps'
        )
    for i, t in enumerate(tqdm(times)) if show_progress else enumerate(times):
        if i < MAX_TIME:
            ds_t = ds.isel({'time' : i}).load()
            ds_t = apply_ortho_map(ds_t, ortho_map, data_vars=[data_var])
            ds_t = smooth(ds_t, input_variable=data_var)

            if running is None:
                running = ds_t[data_var].values
            else:
                running = xr.apply_ufunc(np.fmax, running, ds_t[data_var], dask='allowed')

            s_t = float(np.max(running))
            s_t = s_t if s_t > 0.1 else 0.0
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
        else:
            s_t = 1.0
        
        scaling_factors[i] = s_t

    return scaling_factors

