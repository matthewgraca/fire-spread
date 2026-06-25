# NOTE port over mask-to-confidence mapping code here
# this will be: merge -> filter -> map
# another file will handle the compositing
# remember to fill empty gaps; does Dataset have a ffill/bfill of zeroes?

# remember to filter by top-2 qualities
high_quality_fdc_flags = [
    'good_quality_fire_pixel_qf' ,
    'good_quality_fire_free_land_pixel_qf' ,
    #'invalid_due_to_opaque_cloud_pixel_qf' ,
    #'invalid_due_to_surface_type_or_sunglint_or_LZA_threshold_exceeded_or_off_earth_or_missing_input_data_qf' ,
    #'invalid_due_to_bad_input_data_qf' ,
    #'invalid_due_to_algorithm_failure_qf' ,
]
def add_mask_confidence(scene):
    '''
    Adds a variable to the Dataset converting Mask codes to condfidence values.
        - Mask code meanings: https://cimss.ssec.wisc.edu/satellite-blog/wp-content/uploads/sites/5/2022/03/WFABBA_Mask_Codes_v6.5.012g_2018143_DRAFTwithAWIPSremaps.pdf
        - Confidence values selected according to this scheme: https://essd.copernicus.org/articles/16/1395/2024/#&gid=1&pid=1

    Returns the Dataset with the variable MaskConfidence.
    '''
    if 'Mask' not in scene:
        raise ValueError(
            f"Scene does not contain \"Mask\". "
            "Is this the GOES FDC product, and did you load Mask?"
        )

    mask_confidence = {
        10: 1.0, 30: 1.0,
        11: 0.9, 31: 0.9,
        12: 0.8, 32: 0.8,
        13: 0.5, 33: 0.5,
        14: 0.3, 34: 0.3,
        15: 0.1, 35: 0.1,
    }

    mask = scene['Mask']

    # map mask codes to confidence
    # captures nans from dqf filtering, sets them to 0
    conf = xr.zeros_like(mask, dtype="float32")
    for mask_value, confidence in mask_confidence.items():
        conf = xr.where(mask == mask_value, confidence, conf)

    # metadata for MaskConfidence
    conf.name = "MaskConfidence"
    conf.attrs.update({
        "long_name": "Fire mask confidence",
        "description": (
            "Confidence score derived from GOES FDC Mask values."
        ),
        "valid_min": 0.0,
        "valid_max": 1.0,
    })

    scene["MaskConfidence"] = conf
    return scene

def combine_dataset(stack, save_path=None):
    '''
    Takes a list of DataArrays, and combines them into a coherent Dataset 
    along the time dimension.

    Assumes each element is a DataArray with observations from both satellites.
    '''
    combined = xr.concat(
        stack,
        dim=xr.DataArray(dates, dims='time', name='time')
    )

    combined_ds = xr.DataArray(
        data=combined.data,
        dims=combined.dims,
        coords={
            "time": combined["time"],
            "satellite": combined["satellite"],
            "y": combined["y"],
            "x": combined["x"],
        },
        name="MaskConfidence",
        attrs={
            "long_name": "GOES West and East fire mask confidence",
            "description": "Geolocated confidence scores derived from GOES FDC Mask values.",
            "valid_min": 0.0,
            "valid_max": 1.0,
        },
    ).to_dataset()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        combined_ds.to_netcdf(
            save_path,
            engine="netcdf4",
            encoding={
                "MaskConfidence": {
                    "zlib": True,
                    "complevel": 4,
                    "dtype": "float32",
                    "_FillValue": np.float32(np.nan),
                }
            }
        )
        combined_ds.to_netcdf(save_path)
    return combined_ds
