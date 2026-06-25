# Prunes the FDC product's four variables down to only Mask, and maps Mask values to confidence values based on mask code and data quality

from __future__ import annotations
from typing import Iterable
import numpy as np
import xarray as xr


MASK_CONFIDENCE_MAPPING = {
    10: 1.0,
    30: 1.0,   # Processed / Temporally Filtered

    11: 0.9,
    31: 0.9,   # Saturated / Temporally Filtered

    12: 0.8,
    32: 0.8,   # Cloud contaminated / Temporally Filtered

    13: 0.5,
    33: 0.5,   # High Probability / Temporally Filtered

    14: 0.3,
    34: 0.3,   # Medium Probability / Temporally Filtered

    15: 0.1,
    35: 0.1,   # Low Probability / Temporally Filtered
}


HIGH_QUALITY_FDC_FLAGS = [
    "good_quality_fire_pixel_qf",
    "good_quality_fire_free_land_pixel_qf",
    #'invalid_due_to_opaque_cloud_pixel_qf',
    #'invalid_due_to_surface_type_or_sunglint_or_LZA_threshold_exceeded_or_off_earth_or_missing_input_data_qf',
    #'invalid_due_to_bad_input_data_qf',
    #'invalid_due_to_algorithm_failure_qf' ,
]


FDC_VARIABLES_TO_DROP = [
    "Mask",
    "Temperature",
    "Temp",
    "Power",
    "Area",
]


def _get_quality_flag_values(
    dqf: xr.DataArray,
    desired_flag_meanings: Iterable[str],
) -> list[int]:
    """
    Return DQF integer values corresponding to desired flag meanings.

    Method is primarily seeing if our hardcoded quality flags match the 
    quality flags we're seeing in the nc file. If you trust the nc file 
    completely, you can just use the flags 0 and 1 corresponding to top-2 
    qualities.

    GOES Level 2 products commonly store quality-flag metadata using
    `flag_values` and `flag_meanings` attributes.
    """
    desired = set(desired_flag_meanings)

    flag_meanings = dqf.attrs.get("flag_meanings")
    flag_values = dqf.attrs.get("flag_values")

    if flag_meanings is None or flag_values is None:
        raise ValueError(
            "DQF is missing `flag_meanings` or `flag_values` metadata. "
            "Cannot safely identify high-quality pixels."
        )

    if isinstance(flag_meanings, str):
        meanings = flag_meanings.split()
    else:
        meanings = list(flag_meanings)

    values = np.asarray(flag_values).astype(int).tolist()

    if len(meanings) != len(values):
        raise ValueError(
            "DQF metadata mismatch: `flag_meanings` and `flag_values` "
            "have different lengths."
        )

    mapping = dict(zip(meanings, values))

    missing = [m for m in desired if m not in mapping]
    if missing:
        raise ValueError(
            "Requested DQF flag meanings are missing from the file metadata: "
            + ", ".join(missing)
        )

    return [mapping[m] for m in desired]


def _map_mask_to_confidence(mask: xr.DataArray) -> xr.DataArray:
    """
    Convert GOES FDC Mask codes to continuous confidence values.

    Any unmapped Mask value becomes 0.
    """
    conf = xr.zeros_like(mask, dtype="float32")

    for mask_code, confidence in MASK_CONFIDENCE_MAPPING.items():
        conf = xr.where(mask == mask_code, np.float32(confidence), conf)

    conf.name = "MaskConfidence"

    conf.attrs.update(
        {
            "long_name": "Fire mask confidence",
            "description": (
                "Confidence score derived from GOES FDC Mask values."
            ),
            "valid_min": 0.0,
            "valid_max": 1.0,
            "source_variable": "Mask",
            "confidence_mapping": str(MASK_CONFIDENCE_MAPPING),
            "unmapped_mask_values": "0.0",
        }
    )

    return conf


def fdc_mask_confidence_dataset(
    goes_filepath: str,
    *,
    high_quality_fdc_flags: Iterable[str] = HIGH_QUALITY_FDC_FLAGS,
    variables_to_drop: Iterable[str] = FDC_VARIABLES_TO_DROP,
    load: bool = True,
) -> xr.Dataset:
    """
    Open a GOES FDC/FDCC file and replace fire retrieval variables with
    a single `MaskConfidence` variable.

    Parameters
    ----------
    goes_filepath:
        Path to a GOES FDC/FDCC netCDF file.

    high_quality_fdc_flags:
        DQF flag meanings that are allowed to retain nonzero confidence.
        Pixels whose DQF value is not one of these are assigned confidence 0.

    variables_to_drop:
        Variables to remove from the returned Dataset. By default this removes
        Mask, Temperature, Temp, Power, and Area if present.

    load:
        If True, load the returned Dataset into memory before closing the file.

    Returns
    -------
    xr.Dataset
        Dataset with `Mask`, `Temperature`/`Temp`, `Power`, and `Area` removed,
        and a new `MaskConfidence` variable added.
    """
    with xr.open_dataset(goes_filepath, decode_times=False) as ds:
        if "Mask" not in ds:
            raise ValueError("Dataset is missing required variable `Mask`.")

        if "DQF" not in ds:
            raise ValueError("Dataset is missing required variable `DQF`.")

        high_quality_values = _get_quality_flag_values(
            ds["DQF"],
            high_quality_fdc_flags,
        )

        conf = _map_mask_to_confidence(ds["Mask"])

        good_quality = ds["DQF"].isin(high_quality_values)
        conf = conf.where(good_quality, conf, np.float32(0.0))

        conf.attrs.update(
            {
                "quality_filter": (
                    "MaskConfidence set to 0 where DQF is not one of: "
                    + ", ".join(high_quality_fdc_flags)
                ),
                "allowed_dqf_values": str(high_quality_values),
            }
        )

        drop_names = [name for name in variables_to_drop if name in ds.data_vars]

        out = ds.drop_vars(drop_names)
        out["MaskConfidence"] = conf

        out.attrs.update(ds.attrs)
        out.attrs.update(
            {
                "mask_confidence_created": "true",
                "mask_confidence_source_file": goes_filepath,
                "mask_confidence_removed_variables": ", ".join(drop_names),
                "mask_confidence_quality_flags": ", ".join(high_quality_fdc_flags),
            }
        )

        if load:
            out = out.load()

        return out
