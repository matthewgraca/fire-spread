"""
GOFER fire metrics: active fire line and fire spread rate.

Implements the metrics described in:
    Liu et al. (2024). "Systematically tracking the hourly progression of
    large wildfires using GOES satellite observations." ESSD, 16, 1395–1424.
    https://doi.org/10.5194/essd-16-1395-2024

Metrics computed from a cumulative binary fire perimeter raster (netCDF):
    - fline_r: Retrospective active fire line length (km)
    - fline_c: Concurrent active fire line length (km) — requires active fire
               confidence as a companion variable or separate dataset
    - fspread_mae: Maximum axis of expansion (km/h)
    - fspread_awe: Area-weighted expansion (km/h)

The input netCDF is expected to contain a binary variable (0/1) representing
the cumulative fire perimeter at each hourly timestep, with dimensions
(time, latitude, longitude).

For large fires, use `compute_metrics_streaming(path)` which reads batches
of time slices from disk to avoid OOM.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    center_of_mass,
    distance_transform_edt,
    generate_binary_structure,
)

from gofer.goes_utils import estimate_pixel_size_m


# ---------------------------------------------------------------------------
# Boundary / perimeter extraction
# ---------------------------------------------------------------------------

def _perimeter_mask(mask: np.ndarray) -> np.ndarray:
    """
    Extract the outer boundary of a binary mask.

    A boundary pixel is a fire pixel (mask=1) that has at least one
    non-fire 4-connected neighbor.

    Returns a boolean array of the same shape.
    """
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    struct = generate_binary_structure(2, 1)  # 4-connected cross
    eroded = binary_erosion(mask, structure=struct)
    return mask & ~eroded


def _fire_line_r_mask(perimeter: np.ndarray, new_area: np.ndarray) -> np.ndarray:
    """
    Retrospective fire line: perimeter pixels of the current burned area
    that are adjacent (8-connected) to the new growth region.
    """
    if not new_area.any():
        return np.zeros_like(perimeter, dtype=bool)
    struct = generate_binary_structure(2, 2)  # 8-connected
    new_area_dilated = binary_dilation(new_area, structure=struct)
    return perimeter & new_area_dilated


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _first_fire_timestep(data: np.ndarray) -> int:
    """Find the first timestep index with any burned pixels."""
    for t in range(data.shape[0]):
        if data[t].any():
            return t
    return 0


def _point_mask(shape: tuple, point: tuple) -> np.ndarray:
    """Create a boolean mask with a single True pixel at the given (row, col)."""
    mask = np.zeros(shape, dtype=bool)
    r = int(np.clip(round(point[0]), 0, shape[0] - 1))
    c = int(np.clip(round(point[1]), 0, shape[1] - 1))
    mask[r, c] = True
    return mask


def _decode_batch(raw: np.ndarray, scale: float = 0.01, threshold: float = 0.95) -> np.ndarray:
    """Decode a batch of raw int8 slices to boolean masks (fire/no-fire)."""
    return (raw * scale) >= threshold


# ---------------------------------------------------------------------------
# Active fire line: retrospective (in-memory)
# ---------------------------------------------------------------------------

def fline_r(
    ds: xr.Dataset,
    data_var: str = "MaskConfidence",
) -> xr.DataArray:
    """
    Compute the retrospective active fire line length for each timestep.

    The retrospective fire line at time t consists of the perimeter pixels
    of the burned area at t that are adjacent to new growth at t+1.

    For the last timestep, fline_r = 0 (no future growth observable).

    Args:
        ds: Dataset with a binary (0/1) cumulative perimeter variable.
        data_var: Name of the binary variable.

    Returns:
        DataArray with dimension (time,) containing fline_r in km.
    """
    data = ds[data_var].values.astype(bool)  # (T, H, W)
    n_times = data.shape[0]

    pixel_h, pixel_w = estimate_pixel_size_m(ds)
    pixel_size_m = np.sqrt(pixel_h * pixel_w)

    fline_lengths = np.zeros(n_times, dtype=np.float64)

    for t in range(n_times - 1):
        mask_t = data[t]
        mask_t1 = data[t + 1]

        new_area = mask_t1 & ~mask_t
        if not new_area.any():
            continue

        perimeter = _perimeter_mask(mask_t)
        fline_mask = _fire_line_r_mask(perimeter, new_area)
        fline_lengths[t] = np.sum(fline_mask) * pixel_size_m / 1000.0  # km

    return xr.DataArray(
        data=fline_lengths,
        dims=("time",),
        coords={"time": ds["time"]},
        attrs={"units": "km", "long_name": "Retrospective active fire line length"},
    )


# ---------------------------------------------------------------------------
# Active fire line: concurrent (in-memory)
# ---------------------------------------------------------------------------

def fline_c(
    ds: xr.Dataset,
    confidence: xr.DataArray | xr.Dataset | None = None,
    confidence_var: str = "ActiveFireConfidence",
    confidence_threshold: float = 0.05,
    data_var: str = "MaskConfidence",
) -> xr.DataArray:
    """
    Compute the concurrent active fire line length for each timestep.

    The concurrent fire line identifies which perimeter pixels are actively
    burning at each timestep, based on GOES fire detection confidence.

    This requires the instantaneous (non-cumulative) active fire confidence
    at each hour. This may be:
        - A companion variable in the same dataset (e.g., 'ActiveFireConfidence')
        - A separate xr.Dataset or xr.DataArray passed via `confidence`

    If no confidence data is available, raises NotImplementedError.

    Args:
        ds: Dataset with a binary (0/1) cumulative perimeter variable.
        confidence: Active fire confidence data (0-1) with matching
            (time, latitude, longitude) coordinates. If None, attempts to
            read `confidence_var` from ds. If neither is available, raises.
        confidence_var: Variable name for fire confidence. Used when
            reading from ds or from a Dataset passed as `confidence`.
        confidence_threshold: Minimum confidence to classify a perimeter
            pixel as actively burning. Default 0.05 per Liu et al. (2024).
        data_var: Name of the binary perimeter variable in ds.

    Returns:
        DataArray with dimension (time,) containing fline_c in km.

    Raises:
        NotImplementedError: If no confidence data is available.
    """
    # Resolve confidence data source
    if confidence is not None:
        if isinstance(confidence, xr.Dataset):
            conf_data = confidence[confidence_var].values
        elif isinstance(confidence, xr.DataArray):
            conf_data = confidence.values
        elif isinstance(confidence, np.ndarray):
            conf_data = confidence
        else:
            conf_data = np.asarray(confidence)
    elif confidence_var in ds.data_vars:
        conf_data = ds[confidence_var].values
    else:
        raise NotImplementedError(
            "fline_c requires concurrent GOES fire detection confidence. "
            "The binary perimeter variable alone does not encode which pixels "
            "are actively burning at each timestep.\n\n"
            "To compute fline_c, either:\n"
            "  1. Include an 'ActiveFireConfidence' variable in the dataset "
            "(non-cumulative, hourly max confidence per pixel), or\n"
            "  2. Pass a separate dataset/DataArray via the `confidence` arg.\n\n"
            "In the pipeline, this is produced by running aggregate() with "
            "is_perimeter=False, then applying ortho + smoothing."
        )

    data = ds[data_var].values.astype(bool)  # (T, H, W)
    n_times = data.shape[0]

    pixel_h, pixel_w = estimate_pixel_size_m(ds)
    pixel_size_m = np.sqrt(pixel_h * pixel_w)

    fline_lengths = np.zeros(n_times, dtype=np.float64)

    for t in range(n_times):
        mask_t = data[t]
        perimeter = _perimeter_mask(mask_t)

        if not perimeter.any():
            continue

        active = perimeter & (conf_data[t] >= confidence_threshold)
        fline_lengths[t] = np.sum(active) * pixel_size_m / 1000.0  # km

    return xr.DataArray(
        data=fline_lengths,
        dims=("time",),
        coords={"time": ds["time"]},
        attrs={
            "units": "km",
            "long_name": "Concurrent active fire line length",
            "confidence_threshold": confidence_threshold,
        },
    )


# ---------------------------------------------------------------------------
# Fire spread rate: maximum axis of expansion (in-memory)
# ---------------------------------------------------------------------------

def fspread_mae(
    ds: xr.Dataset,
    data_var: str = "MaskConfidence",
) -> xr.DataArray:
    """
    Compute the maximum axis of expansion (MAE) fire spread rate.

    For each pair of consecutive perimeters (t, t+1), MAE is the maximum
    Euclidean distance from the perimeter at t to any pixel in the new
    burned area (area_{t+1} \\ area_t).

    Special case t=0.5: The "previous perimeter" is the centroid of the
    first observed burned area, and fspread_mae is the maximum distance
    from that centroid to any pixel in the first burned area.

    Results are indexed at half-hour timesteps (t+0.5), following the GOFER
    convention where spread rates represent transitions between consecutive
    end-of-hour perimeters.

    Args:
        ds: Dataset with a binary (0/1) cumulative perimeter variable.
        data_var: Name of the binary variable.

    Returns:
        DataArray with dimension (time_mid,) containing fspread_mae in km/h.
    """
    data = ds[data_var].values.astype(bool)  # (T, H, W)
    n_times = data.shape[0]
    times = ds["time"].values

    pixel_h, pixel_w = estimate_pixel_size_m(ds)
    spacing = (pixel_h, pixel_w)  # (row_spacing, col_spacing) in meters

    # Find first timestep with any burned area
    first_t = _first_fire_timestep(data)

    # Number of transitions: first_t→first_t is t=0.5, then first_t→first_t+1, etc.
    n_transitions = n_times - first_t
    mae_values = np.zeros(n_transitions, dtype=np.float64)
    mid_times = np.empty(n_transitions, dtype="datetime64[ns]")

    # --- Special case: t=0.5 (centroid → first perimeter) ---
    mask_first = data[first_t]
    centroid = center_of_mass(mask_first)  # (row, col) fractional
    origin_mask = _point_mask(mask_first.shape, centroid)
    dist_from_centroid = distance_transform_edt(~origin_mask, sampling=spacing)

    mae_values[0] = dist_from_centroid[mask_first].max() / 1000.0  # km/h

    # Mid-time for t=0.5
    if first_t + 1 < n_times:
        dt = (times[first_t + 1] - times[first_t]) / 2
    else:
        dt = np.timedelta64(30, "m")
    mid_times[0] = times[first_t] - dt  # half-step before first_t

    # --- Normal case: transitions from first_t onward ---
    for i in range(1, n_transitions):
        t = first_t + i - 1
        t1 = t + 1

        mask_t = data[t]
        mask_t1 = data[t1]
        new_area = mask_t1 & ~mask_t

        mid_times[i] = times[t] + (times[t1] - times[t]) / 2

        if not new_area.any():
            mae_values[i] = 0.0
            continue

        if not mask_t.any():
            # Edge case: no previous area, use centroid of new area
            c = center_of_mass(new_area)
            origin = _point_mask(mask_t.shape, c)
            dist = distance_transform_edt(~origin, sampling=spacing)
            mae_values[i] = dist[new_area].max() / 1000.0
            continue

        # Distance transform from perimeter at t
        perimeter = _perimeter_mask(mask_t)
        dist_from_perimeter = distance_transform_edt(~perimeter, sampling=spacing)
        mae_values[i] = dist_from_perimeter[new_area].max() / 1000.0  # km/h

    return xr.DataArray(
        data=mae_values,
        dims=("time_mid",),
        coords={"time_mid": mid_times},
        attrs={
            "units": "km/h",
            "long_name": "Fire spread rate — maximum axis of expansion",
        },
    )


# ---------------------------------------------------------------------------
# Fire spread rate: area-weighted expansion (in-memory)
# ---------------------------------------------------------------------------

def fspread_awe(
    ds: xr.Dataset,
    data_var: str = "MaskConfidence",
    fline_r_da: xr.DataArray | None = None,
) -> xr.DataArray:
    """
    Compute the area-weighted expansion (AWE) fire spread rate.

    For t > first timestep:
        fspread_awe = dArea / fline_r(t)

    Special case t=0.5:
        fspread_awe = mean distance from centroid of first perimeter to the
        boundary pixels of that perimeter.

    When fline_r = 0 but growth exists (spotting/new ignition disconnected
    from the main fire), falls back to mean distance from the centroid of
    the new growth to its boundary.

    Args:
        ds: Dataset with a binary (0/1) cumulative perimeter variable.
        data_var: Name of the binary variable.
        fline_r_da: Pre-computed fline_r DataArray. If None, computed here.

    Returns:
        DataArray with dimension (time_mid,) containing fspread_awe in km/h.
    """
    data = ds[data_var].values.astype(bool)  # (T, H, W)
    n_times = data.shape[0]
    times = ds["time"].values

    pixel_h, pixel_w = estimate_pixel_size_m(ds)
    spacing = (pixel_h, pixel_w)
    pixel_area_km2 = (pixel_h * pixel_w) / 1e6

    # Get fline_r
    if fline_r_da is None:
        fline_r_da = fline_r(ds, data_var=data_var)
    fline_r_values = fline_r_da.values  # km, indexed by time

    first_t = _first_fire_timestep(data)
    n_transitions = n_times - first_t
    awe_values = np.zeros(n_transitions, dtype=np.float64)
    mid_times = np.empty(n_transitions, dtype="datetime64[ns]")

    # --- Special case t=0.5: mean distance from centroid to boundary ---
    mask_first = data[first_t]
    centroid = center_of_mass(mask_first)
    origin_mask = _point_mask(mask_first.shape, centroid)
    dist_from_centroid = distance_transform_edt(~origin_mask, sampling=spacing)

    boundary_first = _perimeter_mask(mask_first)
    if boundary_first.any():
        awe_values[0] = dist_from_centroid[boundary_first].mean() / 1000.0
    else:
        # Single pixel — distance is 0
        awe_values[0] = 0.0

    if first_t + 1 < n_times:
        dt = (times[first_t + 1] - times[first_t]) / 2
    else:
        dt = np.timedelta64(30, "m")
    mid_times[0] = times[first_t] - dt

    # --- Normal case: dArea / fline_r ---
    for i in range(1, n_transitions):
        t = first_t + i - 1
        t1 = t + 1

        mask_t = data[t]
        mask_t1 = data[t1]
        new_area = mask_t1 & ~mask_t

        mid_times[i] = times[t] + (times[t1] - times[t]) / 2

        if not new_area.any():
            awe_values[i] = 0.0
            continue

        dArea_km2 = float(np.sum(new_area)) * pixel_area_km2
        fl = fline_r_values[t]  # fline_r at time t (km)

        if fl > 0:
            awe_values[i] = dArea_km2 / fl  # km²/km = km/h
        else:
            # No fire line but growth exists — fallback to centroid method
            c = center_of_mass(new_area)
            origin = _point_mask(mask_t.shape, c)
            dist = distance_transform_edt(~origin, sampling=spacing)
            boundary = _perimeter_mask(new_area)
            if boundary.any():
                awe_values[i] = dist[boundary].mean() / 1000.0
            else:
                awe_values[i] = dist[new_area].mean() / 1000.0

    return xr.DataArray(
        data=awe_values,
        dims=("time_mid",),
        coords={"time_mid": mid_times},
        attrs={
            "units": "km/h",
            "long_name": "Fire spread rate — area-weighted expansion",
        },
    )


# ---------------------------------------------------------------------------
# Streaming metrics computation (batched reads from disk)
# ---------------------------------------------------------------------------

def compute_metrics_streaming(
    nc_path: str | Path,
    data_var: str = "MaskConfidence",
    confidence_var: str = "ActiveFireConfidence",
    confidence_threshold: float = 0.05,
    batch_size: int = 200,
) -> xr.Dataset:
    """
    Compute all fire metrics by streaming batches of time slices from disk.

    This avoids loading the entire (T, H, W) array into memory. Instead,
    reads `batch_size` slices at a time — keeping at most 2 batches resident
    (current + overlap from previous). For a 1500×2500 grid with batch_size=200,
    peak memory is ~200 × 1500 × 2500 × 1 byte ≈ 750 MB per variable.

    Computes:
        - fline_r (time): Retrospective fire line length (km)
        - fspread_mae (time_mid): Maximum axis of expansion (km/h)
        - fspread_awe (time_mid): Area-weighted expansion (km/h)
        - fline_c (time): Concurrent fire line length (km) — if confidence
          variable is present in the file.

    Args:
        nc_path: Path to the final GOFER netCDF file.
        data_var: Name of the binary perimeter variable (encoded as int8).
        confidence_var: Name of the active fire confidence variable.
        confidence_threshold: Confidence cutoff for fline_c.
        batch_size: Number of time slices to read per batch. Higher values
            use more memory but reduce I/O overhead. Default 200.

    Returns:
        Dataset with the computed metrics as 1D variables.
    """
    import h5py

    nc_path = str(nc_path)

    with h5py.File(nc_path, 'r') as f:
        mc = f[data_var]
        n_times, n_lat, n_lon = mc.shape

        has_confidence = confidence_var in f
        afc = f[confidence_var] if has_confidence else None

        # Read coordinates
        lat_vals = f['latitude'][:]
        lon_vals = f['longitude'][:]

        # Decode time: stored as minutes since epoch
        time_raw = f['time'][:]
        time_vals = (time_raw.astype('int64') * 60_000_000_000).astype('datetime64[ns]')

        # Compute pixel geometry using estimate_pixel_size_m via a minimal Dataset
        _ds = xr.Dataset(coords={
            "latitude": ("latitude", lat_vals),
            "longitude": ("longitude", lon_vals),
        })
        pixel_h, pixel_w = estimate_pixel_size_m(_ds)
        pixel_size_m = np.sqrt(pixel_h * pixel_w)
        spacing = (pixel_h, pixel_w)
        pixel_area_km2 = (pixel_h * pixel_w) / 1e6

        # --- Pass 1: find first fire timestep (scan single slices) ---
        first_t = 0
        for t in range(n_times):
            raw_slice = mc[t]
            if np.any(_decode_batch(raw_slice)):
                first_t = t
                break

        # --- Allocate output arrays ---
        fline_r_values = np.zeros(n_times, dtype=np.float64)
        fline_c_values = np.zeros(n_times, dtype=np.float64) if has_confidence else None

        n_transitions = n_times - first_t
        mae_values = np.zeros(n_transitions, dtype=np.float64)
        awe_values = np.zeros(n_transitions, dtype=np.float64)
        mid_times = np.empty(n_transitions, dtype="datetime64[ns]")

        # --- Special case: t=0.5 (centroid → first perimeter) ---
        mask_first = _decode_batch(mc[first_t])
        centroid = center_of_mass(mask_first)
        origin_mask = _point_mask(mask_first.shape, centroid)
        dist_from_centroid = distance_transform_edt(~origin_mask, sampling=spacing)

        mae_values[0] = dist_from_centroid[mask_first].max() / 1000.0

        boundary_first = _perimeter_mask(mask_first)
        if boundary_first.any():
            awe_values[0] = dist_from_centroid[boundary_first].mean() / 1000.0
        else:
            awe_values[0] = 0.0

        if first_t + 1 < n_times:
            dt = (time_vals[first_t + 1] - time_vals[first_t]) / 2
        else:
            dt = np.timedelta64(30, "m")
        mid_times[0] = time_vals[first_t] - dt

        # --- Special case: fline_c at first_t ---
        if has_confidence:
            perimeter_first = _perimeter_mask(mask_first)
            if perimeter_first.any():
                afc_slice = afc[first_t].astype(np.float32) * np.float32(0.01)
                active = perimeter_first & (afc_slice >= confidence_threshold)
                fline_c_values[first_t] = np.sum(active) * pixel_size_m / 1000.0

        del mask_first, dist_from_centroid, origin_mask, boundary_first

        # --- Main loop: batched reads over transitions ---
        # We process transitions (t, t+1) for t in [first_t, n_times-2].
        # Each batch reads a contiguous block and processes all pairs within it.
        # The last slice of a batch becomes the first slice of the next batch.

        t_start = first_t
        prev_slice = _decode_batch(mc[first_t])

        while t_start < n_times - 1:
            # Read a batch: [t_start+1, t_start+1+batch_size)
            # We already have prev_slice = decoded data[t_start]
            batch_end = min(t_start + 1 + batch_size, n_times)
            batch_raw = mc[t_start + 1:batch_end]
            batch_decoded = _decode_batch(batch_raw)
            del batch_raw

            # Also read confidence batch if needed
            if has_confidence:
                afc_batch = afc[t_start + 1:batch_end].astype(np.float32) * np.float32(0.01)
            else:
                afc_batch = None

            # Process each transition in this batch
            for local_idx in range(batch_decoded.shape[0]):
                t = t_start + local_idx  # absolute time index of "current"
                t1 = t + 1               # absolute time index of "next"

                mask_t = prev_slice
                mask_t1 = batch_decoded[local_idx]

                # Transition index into mae/awe arrays
                i = t - first_t + 1  # offset: i=0 is the special case above

                if i < n_transitions:
                    new_area = mask_t1 & ~mask_t
                    mid_times[i] = time_vals[t] + (time_vals[t1] - time_vals[t]) / 2

                    # --- fline_r ---
                    if new_area.any():
                        perimeter = _perimeter_mask(mask_t)
                        fline_mask = _fire_line_r_mask(perimeter, new_area)
                        fline_r_values[t] = np.sum(fline_mask) * pixel_size_m / 1000.0

                        # --- fspread_mae ---
                        dist_from_perimeter = distance_transform_edt(
                            ~perimeter, sampling=spacing
                        )
                        mae_values[i] = dist_from_perimeter[new_area].max() / 1000.0

                        # --- fspread_awe ---
                        dArea_km2 = float(np.sum(new_area)) * pixel_area_km2
                        fl = fline_r_values[t]
                        if fl > 0:
                            awe_values[i] = dArea_km2 / fl
                        else:
                            c = center_of_mass(new_area)
                            origin = _point_mask(mask_t.shape, c)
                            dist = distance_transform_edt(~origin, sampling=spacing)
                            boundary = _perimeter_mask(new_area)
                            if boundary.any():
                                awe_values[i] = dist[boundary].mean() / 1000.0
                            else:
                                awe_values[i] = dist[new_area].mean() / 1000.0
                    else:
                        mae_values[i] = 0.0
                        awe_values[i] = 0.0

                # --- fline_c for t1 ---
                if has_confidence:
                    perimeter_t1 = _perimeter_mask(mask_t1)
                    if perimeter_t1.any():
                        active = perimeter_t1 & (afc_batch[local_idx] >= confidence_threshold)
                        fline_c_values[t1] = np.sum(active) * pixel_size_m / 1000.0

                # Also handle the edge case: t < first_t transitions skipped
                # but fline_c still needs perimeter at t1
                # (already handled above since we iterate all t1 in batch)

                prev_slice = mask_t1

            t_start = batch_end - 1
            del batch_decoded, afc_batch

        # Handle edge case: mask_t has no fire but new_area exists
        # (already handled in the loop — falls through to centroid method)

    # --- Build result Dataset ---
    result_vars = {
        "fline_r": xr.DataArray(
            data=fline_r_values,
            dims=("time",),
            coords={"time": time_vals},
            attrs={"units": "km", "long_name": "Retrospective active fire line length"},
        ),
        "fspread_mae": xr.DataArray(
            data=mae_values,
            dims=("time_mid",),
            coords={"time_mid": mid_times},
            attrs={"units": "km/h", "long_name": "Fire spread rate — maximum axis of expansion"},
        ),
        "fspread_awe": xr.DataArray(
            data=awe_values,
            dims=("time_mid",),
            coords={"time_mid": mid_times},
            attrs={"units": "km/h", "long_name": "Fire spread rate — area-weighted expansion"},
        ),
    }

    if has_confidence:
        result_vars["fline_c"] = xr.DataArray(
            data=fline_c_values,
            dims=("time",),
            coords={"time": time_vals},
            attrs={
                "units": "km",
                "long_name": "Concurrent active fire line length",
                "confidence_threshold": confidence_threshold,
            },
        )

    return xr.Dataset(
        result_vars,
        attrs={
            "description": (
                "GOFER fire metrics computed from binary perimeter raster. "
                "See Liu et al. (2024) ESSD 16:1395-1424."
            ),
        },
    )


# ---------------------------------------------------------------------------
# Convenience: compute all metrics at once (in-memory, for small fires)
# ---------------------------------------------------------------------------

def compute_metrics(
    ds: xr.Dataset,
    data_var: str = "MaskConfidence",
    confidence_var: str | None = None,
    confidence_threshold: float = 0.05,
) -> xr.Dataset:
    """
    Compute all available fire metrics from a binary perimeter netCDF.

    NOTE: This loads the entire dataset into memory. For large fires
    (>1000 timesteps or large spatial grids), use compute_metrics_streaming()
    instead.

    Always computes:
        - fline_r (time): Retrospective fire line length (km)
        - fspread_mae (time_mid): Maximum axis of expansion (km/h)
        - fspread_awe (time_mid): Area-weighted expansion (km/h)

    Optionally computes (if active fire confidence is available):
        - fline_c (time): Concurrent fire line length (km)

    The dataset is cummax-enforced before computation to guarantee
    monotonically non-decreasing perimeters.

    Args:
        ds: Dataset with a binary (0/1) cumulative perimeter variable.
        data_var: Name of the binary perimeter variable.
        confidence_var: If present in ds, used to compute fline_c.
            Set to None to skip fline_c. Default checks for
            'ActiveFireConfidence'.
        confidence_threshold: Confidence cutoff for fline_c.

    Returns:
        Dataset with the computed metrics.
    """
    from gofer.postprocess import enforce_cummax

    ds = enforce_cummax(ds, data_var=data_var)

    fl_r = fline_r(ds, data_var=data_var)
    mae = fspread_mae(ds, data_var=data_var)
    awe = fspread_awe(ds, data_var=data_var, fline_r_da=fl_r)

    result_vars = {
        "fline_r": fl_r,
        "fspread_mae": mae,
        "fspread_awe": awe,
    }

    # Attempt fline_c if confidence data is available
    if confidence_var is None:
        confidence_var = "ActiveFireConfidence"
    if confidence_var in ds.data_vars:
        fl_c = fline_c(
            ds,
            confidence_var=confidence_var,
            confidence_threshold=confidence_threshold,
            data_var=data_var,
        )
        result_vars["fline_c"] = fl_c

    return xr.Dataset(
        result_vars,
        attrs={
            "description": (
                "GOFER fire metrics computed from binary perimeter raster. "
                "See Liu et al. (2024) ESSD 16:1395-1424."
            ),
        },
    )
