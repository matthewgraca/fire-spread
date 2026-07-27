"""
Generate synthetic test data for temporal_downsampler tests.

Creates 8 NC files (2 per hour, 4 hours with data + 1 hour gap) and a CSV inventory.
Spatial dimensions are 10x10. Files mimic the GOES FDC product structure.

Timeline:
    Hour 1 (20:00): 2 observations -> fire starts (2x2)
    Hour 2 (21:00): 2 observations -> fire grows (3x3)
    Hour 3 (22:00): GAP - no data (tests imputation)
    Hour 4 (23:00): 2 observations -> fire grows (4x4)
    Hour 5 (00:00): 2 observations -> fire unchanged (4x4, tests trimming)

Run this once to create the test fixtures:
    python tests/gofer/generate_temporal_test_data.py
"""
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("tests/gofer/data/temporal")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Reference time for GOES 't' variable: seconds since 2000-01-01 12:00:00
ORIGIN = pd.Timestamp("2000-01-01 12:00:00")

# 4 hours of data, 2 observations per hour, with a gap at hour 3
# Hour 1: 2020-09-06 20:00 (observations at 19:05, 19:35)
# Hour 2: 2020-09-06 21:00 (observations at 20:10, 20:40)
# Hour 3: 2020-09-06 22:00 -- NO DATA (gap)
# Hour 4: 2020-09-06 23:00 (observations at 22:15, 22:45)
# Hour 5: 2020-09-07 00:00 (observations at 23:20, 23:50)
observation_times = [
    pd.Timestamp("2020-09-06 19:05:00"),
    pd.Timestamp("2020-09-06 19:35:00"),
    pd.Timestamp("2020-09-06 20:10:00"),
    pd.Timestamp("2020-09-06 20:40:00"),
    # gap at hour 3 -- no observations
    pd.Timestamp("2020-09-06 22:15:00"),
    pd.Timestamp("2020-09-06 22:45:00"),
    pd.Timestamp("2020-09-06 23:20:00"),
    pd.Timestamp("2020-09-06 23:50:00"),
]

# DQF attributes matching real GOES data
DQF_ATTRS = {
    "flag_values": np.array([0, 1, 2, 3, 4, 5], dtype=np.int8),
    "flag_meanings": (
        "good_quality_fire_pixel_qf "
        "good_quality_fire_free_land_pixel_qf "
        "invalid_due_to_opaque_cloud_pixel_qf "
        "invalid_due_to_surface_type_or_sunglint_or_LZA_threshold_exceeded_or_off_earth_or_missing_input_data_qf "
        "invalid_due_to_bad_input_data_qf "
        "invalid_due_to_algorithm_failure_qf"
    ),
}


def make_mask(obs_idx: int) -> np.ndarray:
    """Create a 10x10 mask array. Fire grows as obs_idx increases."""
    mask = np.zeros((10, 10), dtype=np.int16)

    if obs_idx in (0, 1):  # Hour 1: small 2x2 fire in center
        mask[4:6, 4:6] = 10
    elif obs_idx in (2, 3):  # Hour 2: grows to 3x3
        mask[4:7, 4:7] = 10
        # One pixel with lower confidence in obs 2, higher in obs 3
        # Tests that max is taken across subhourly observations
        if obs_idx == 2:
            mask[3, 5] = 13  # high probability (0.5)
        else:
            mask[3, 5] = 10  # processed (1.0) -- this should win via max
    elif obs_idx in (4, 5):  # Hour 4 (after gap): grows to 4x4
        mask[3:7, 3:7] = 10
    elif obs_idx in (6, 7):  # Hour 5: same as hour 4 (fire stops growing)
        mask[3:7, 3:7] = 10

    return mask


def make_dqf(obs_idx: int) -> np.ndarray:
    """Create a 10x10 DQF array. All good quality."""
    return np.zeros((10, 10), dtype=np.int8)


# Fake x/y coordinates (ABI scan angles, arbitrary for testing)
y_coords = np.linspace(0.1, 0.11, 10).astype(np.float32)
x_coords = np.linspace(-0.05, -0.04, 10).astype(np.float32)

# Projection variable (minimal, needed for pipeline compatibility)
goes_imager_projection_attrs = {
    "semi_major_axis": 6378137.0,
    "semi_minor_axis": 6356752.31414,
    "perspective_point_height": 35786023.0,
    "longitude_of_projection_origin": -137.0,
}

csv_records = []

for i, obs_time in enumerate(observation_times):
    t_seconds = (obs_time - ORIGIN).total_seconds()

    ds = xr.Dataset(
        data_vars={
            "Mask": xr.DataArray(
                make_mask(i)[np.newaxis, :, :],
                dims=["time", "y", "x"],
            ),
            "DQF": xr.DataArray(
                make_dqf(i)[np.newaxis, :, :],
                dims=["time", "y", "x"],
                attrs=DQF_ATTRS,
            ),
            "t": xr.DataArray(
                np.array([t_seconds], dtype=np.float64),
                dims=["time"],
                attrs={"units": "seconds since 2000-01-01 12:00:00"},
            ),
            "goes_imager_projection": xr.DataArray(
                np.int32(0),
                attrs=goes_imager_projection_attrs,
            ),
        },
        coords={
            "y": y_coords,
            "x": x_coords,
        },
    )

    filename = f"obs_{i:02d}_{obs_time.strftime('%Y%m%d_%H%M')}.nc"
    filepath = OUTPUT_DIR / filename
    ds.to_netcdf(filepath)

    # CSV record mimicking goes2go output
    csv_records.append({
        "file": f"temporal/{filename}",
        "start": obs_time,
        "end": obs_time + pd.Timedelta(minutes=5),
        "creation": obs_time + pd.Timedelta(minutes=2),
    })

# Write CSV
csv_df = pd.DataFrame(csv_records)
csv_df.to_csv(OUTPUT_DIR / "test_files.csv", index=False)

print(f"Created {len(observation_times)} NC files and CSV in {OUTPUT_DIR}")
print(csv_df)
print(f"\nExpected hourly dates (5 hours, with gap at hour 3):")
dates = pd.date_range("2020-09-06 20:00", periods=5, freq="h")
print(dates)
