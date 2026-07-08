"""
Takes in a Dataset

For each hour h:

    Gather the confidences from hour [1, h]
    Calculate the cumulative max of the confidences from [1, h]
    Parallax adjustment
    Apply spatial smoothing
    Calculate the max confidence of the entire frame s_h; the scaling factor (+ exceptions like t >= 500)
    For the entire frame, divide each pixel by s_h

"""
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
    da = ds[data_var]
    times = da['time'].values
    running = None
    scaling_factors = []
    ortho_map = make_ortho_map(ds, **ortho_kwargs)
    for i, t in enumerate(tqdm(times)) if show_progress else enumerate(times):
        frame = da.isel({'time': i}).astype('float32').load()

        if running is None:
            running = frame.copy(deep=True)
        else:
            running = xr.apply_ufunc(np.fmax, running, frame, dask='allowed')

        temp_ds = ds.isel({'time' : i}).load()
        temp_ds[data_var] = running

        temp_ds = apply_ortho_map(temp_ds, ortho_map, data_vars=[data_var])
        temp_ds = smooth(temp_ds, input_variable=data_var)

        s_t = temp_ds[data_var].max(dim=('latitude', 'longitude'), skipna=True).item()

        scaling_factors.append(s_t)

    return scaling_factors

