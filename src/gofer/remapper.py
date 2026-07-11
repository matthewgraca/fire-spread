# Utilities for remapping GOES datasets.

from __future__ import annotations

from collections.abc import Iterable 

import numpy as np
import xarray as xr

#### BEGIN CONSTANTS 

MASK_CONFIDENCE_MAPPING = {
    10: 1.0,
    30: 1.0,  # Processed / Temporally Filtered
    11: 1.0,
    31: 1.0,  # Saturated / Temporally Filtered
    12: 0.8,
    32: 0.8,  # Cloud contaminated / Temporally Filtered
    13: 0.5,
    33: 0.5,  # High Probability / Temporally Filtered
    14: 0.3,
    34: 0.3,  # Medium Probability / Temporally Filtered
    15: 0.1,
    35: 0.1,  # Low Probability / Temporally Filtered
    # any value that doesn't meet top 2 quality will be set to 0.0
}


HIGH_QUALITY_FDC_FLAGS = [
    "good_quality_fire_pixel_qf",
    "good_quality_fire_free_land_pixel_qf",
    # "invalid_due_to_opaque_cloud_pixel_qf",
    # "invalid_due_to_surface_type_or_sunglint_or_LZA_threshold_exceeded_or_off_earth_or_missing_input_data_qf",
    # "invalid_due_to_bad_input_data_qf",
    # "invalid_due_to_algorithm_failure_qf",
]

#### END CONSTANTS

def _get_quality_flag_values(
    dqf: xr.DataArray,
    desired_flag_meanings: Iterable[str],
) -> list[int]:
    """
    Return DQF integer values corresponding to desired flag meanings.

    GOES Level 2 products commonly store quality-flag metadata using
    ``flag_values`` and ``flag_meanings`` attributes.

    You could just trust that 0 and 1 are the top two qualities. However, this 
    lookup keeps the remap tied to file metadata rather than assuming that 
    0 and 1 always mean the desired quality classes.
    """
    desired = list(dict.fromkeys(desired_flag_meanings))

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

    mapping = dict(zip(meanings, values, strict=True))

    missing = [meaning for meaning in desired if meaning not in mapping]
    if missing:
        raise ValueError(
            "Requested DQF flag meanings are missing from the file metadata: "
            + ", ".join(missing)
        )

    return [mapping[meaning] for meaning in desired]


def map_fdc_mask_to_confidence(
    goes_ds: xr.Dataset,
    high_quality_fdc_flags: Iterable[str] = HIGH_QUALITY_FDC_FLAGS,
    confidence_mapping: dict[int, float] = MASK_CONFIDENCE_MAPPING,
) -> xr.Dataset:
    """
    Create ``MaskConfidence`` from FDC ``Mask`` and ``DQF``.

    After ``MaskConfidence`` is created, ``DQF`` is dropped by default because
    its quality information has already been encoded in the confidence score.
    ``Mask`` is also dropped by default because it is only an intermediate input
    for this remap.
    """
    if "Mask" not in goes_ds.data_vars:
        raise ValueError("Dataset is missing required variable `Mask`.")

    if "DQF" not in goes_ds.data_vars:
        raise ValueError("Dataset is missing required variable `DQF`.")

    # replace relevant mask codes with confidence; all other codes will be 0.0
    conf = xr.zeros_like(goes_ds["Mask"], dtype="float32")
    for mask_code, confidence in confidence_mapping.items():
        conf = xr.where(goes_ds["Mask"] == mask_code, np.float32(confidence), conf)

    conf.name = "MaskConfidence" 

    # replace any low quality pixels with 0.0, no exceptions
    high_quality_values = _get_quality_flag_values(
        goes_ds["DQF"],
        high_quality_fdc_flags,
    )

    good_quality = goes_ds["DQF"].isin(high_quality_values)
    conf = conf.where(good_quality, np.float32(0.0))

    # slot in MaskConfidence to the Dataset, while dropping the intermediaries
    out = goes_ds.drop_vars(["Mask", "DQF"])
    out["MaskConfidence"] = conf
    out["MaskConfidence"].attrs.update(
        {
            "long_name": "Fire mask confidence",
            "description": "Confidence score derived from GOES FDC Mask values.",
            "valid_min": 0.0,
            "valid_max": 1.0,
            "confidence_mapping": str(confidence_mapping),
        }
    )

    return out
