'''
Applies a smoothing kernel (boxcar average, moving average, neighborhood mean) 
to processed GOES data.
'''
import numpy as np
import xarray as xr
from scipy.ndimage import uniform_filter
import time


def estimate_pixel_size_m(ds: xr.Dataset) -> tuple[float, float]:
    """
    Since we expect a grid of values, we will need to estimate approximate 
    height and width in meters from latitude/longitude for each pixel

    Assumes regular lat/lon grid.
    """
    lat = ds["latitude"].values
    lon = ds["longitude"].values

    dlat = abs(np.nanmedian(np.diff(lat)))
    dlon = abs(np.nanmedian(np.diff(lon)))

    mean_lat = float(np.nanmean(lat))

    meters_per_degree_lat = 111_320
    meters_per_degree_lon = 111_320 * np.cos(np.deg2rad(mean_lat))

    pixel_height_m = dlat * meters_per_degree_lat
    pixel_width_m = dlon * meters_per_degree_lon

    return pixel_height_m, pixel_width_m


def _kernel_size_from_meters(ds: xr.Dataset, kernel_width_m: float) -> int:
    """
    The size of the kernel changes depending on the size of the pixels. We 
    usually want a given size in meters (like 1700), so for each unique 
    Dataset, we'll need to dynamically determine the kernel size.

    We also force an odd size, since the smoothing is done such that the 
    center pixel takes on the mean of the kernel.

    Convert a desired square kernel width in meters to an odd pixel kernel size.
    """
    pixel_height_m, pixel_width_m = estimate_pixel_size_m(ds)
    nominal_pixel_size_m = np.sqrt(pixel_height_m * pixel_width_m)

    kernel_size = int(round(kernel_width_m / nominal_pixel_size_m))
    
    # Ensure at least 1 and force odd size
    kernel_size = max(kernel_size, 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    return kernel_size

def _get_kernel_dims(da: xr.DataArray, kernel_size: int) -> tuple:
    '''
    Ensures the kernel dimensions only work on the spatial dimensions by
    forcing all non-spatial dimensions to have a kernel size of 1.

    For example, if the da has (time, latitude, longitude), then 
    k = (1, kernel_size, kernel_size)

    Or, if the da has (time, band latitude, longitude), then 
    k = (1, 1, kernel_size, kernel_size)
    '''
    size_by_dim = {
        dim: kernel_size if dim in {"latitude", "longitude"} else 1
        for dim in da.dims
    }

    return tuple(size_by_dim[dim] for dim in da.dims)

def smooth_displacement(
    abi_x: np.ndarray,
    abi_y: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    kernel_radius_m: float = 1700,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Smooth the ABI scan-angle displacement arrays with a neighborhood mean.

    In the original GOFER pipeline, the parallax displacement vectors are
    smoothed with the same spatial kernel before being applied. This prevents
    sharp terrain-scale discontinuities from fragmenting the coarse GOES signal
    during orthorectification.

    Args:
        abi_x: 2D array of ABI x scan angles (radians), shape (lat, lon).
        abi_y: 2D array of ABI y scan angles (radians), shape (lat, lon).
        lon: 1D array of longitude values for the grid.
        lat: 1D array of latitude values for the grid.
        kernel_radius_m: Radius of the smoothing kernel in meters. The full
            kernel width is 2 * kernel_radius_m.

    Returns:
        Tuple of (smoothed_abi_x, smoothed_abi_y).
    """
    # Estimate pixel size from the lat/lon grid
    dlat = abs(np.nanmedian(np.diff(lat)))
    dlon = abs(np.nanmedian(np.diff(lon)))
    mean_lat = float(np.nanmean(lat))

    meters_per_degree_lat = 111_320
    meters_per_degree_lon = 111_320 * np.cos(np.deg2rad(mean_lat))

    pixel_height_m = dlat * meters_per_degree_lat
    pixel_width_m = dlon * meters_per_degree_lon
    nominal_pixel_size_m = np.sqrt(pixel_height_m * pixel_width_m)

    kernel_size = int(round((kernel_radius_m * 2) / nominal_pixel_size_m))
    kernel_size = max(kernel_size, 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    smoothed_x = uniform_filter(abi_x, size=kernel_size, mode="nearest")
    smoothed_y = uniform_filter(abi_y, size=kernel_size, mode="nearest")

    return smoothed_x, smoothed_y


def smooth(
    ds: xr.Dataset,
    kernel_radius_m: int = 1700,
    input_variable: str = "MaskConfidence"
) -> xr.Dataset:
    """
    Smooths a given Dataset variable according to a desired kernel radius 
    in meters. Smoothing is done by taking the mean of the kernel and 
    assigning it to the center of the kernel.

    The effect of this smoothing is mostly on the edges, keeping the body 
    of the structure intact.
    """
    kernel_size = _kernel_size_from_meters(ds, kernel_radius_m * 2)

    da = ds[input_variable]
    smoothed_values = uniform_filter(
        da.values,
        size=_get_kernel_dims(da, kernel_size),
        mode="nearest"
    )

    out = ds.copy()
    out[input_variable] = xr.DataArray(
        smoothed_values,
        dims=da.dims,
        coords=da.coords
    )

    out = out.assign_attrs(pipeline='smoothed')

    return out
