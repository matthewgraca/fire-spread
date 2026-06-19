import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt
import pickle
from tqdm import tqdm
from satpy import Scene
from satpy.area import get_area_def
from pyresample import create_area_def
import logging
import numpy as np
from .tilers import CartoDBTiles

logging.basicConfig(level=logging.ERROR)

def plot(scn_west, scn_east, scn_merged, date, i, save_dir):
    fig, axes = plt.subplots(
        1, 3,
        figsize=(16, 6),
        subplot_kw={'projection' : ccrs.PlateCarree()},
        layout='constrained'
    )

    # base map
    tiler = CartoDBTiles(style='rastertiles/voyager', cache=True)
    for ax in axes:
        ax.add_image(tiler, 12)
        ax.set_extent(extent, crs=ccrs.PlateCarree())

    # data plots
    plot_shared_kwargs = {
        'transform' : ccrs.PlateCarree(),
        'cmap' : 'YlOrRd',
        'vmin' : 0.0,
        'vmax' : 1.0,
        'add_colorbar' : False
    }

    west_data = scn_west['MaskConfidence'].where(scn_west['MaskConfidence'] != 0)
    east_data = scn_east['MaskConfidence'].where(scn_east['MaskConfidence'] != 0)
    merged_data = scn_merged.where(scn_merged != 0)

    west_plot = west_data.plot(ax=axes[0], **plot_shared_kwargs)
    east_plot = east_data.plot(ax=axes[1], **plot_shared_kwargs)
    merged_plot = merged_data.plot(ax=axes[2], **plot_shared_kwargs)

    # colorbar
    cbar = fig.colorbar(
        west_plot,
        ax=axes,
        orientation='vertical',
        shrink=0.7,
        pad=0.03
    )
    cbar.set_label("Fire Confidence")

    # titles
    axes[0].set_title(
        f'GOES-West'
        f'\nstart: {scn_west.start_time.strftime("%Y-%m-%d %H:%M:%S")}'
        f'\nend: {scn_west.end_time.strftime("%Y-%m-%d %H:%M:%S")}'
    )
    axes[1].set_title(
        f'GOES-East'
        f'\nstart: {scn_east.start_time.strftime("%Y-%m-%d %H:%M:%S")}'
        f'\nend: {scn_east.end_time.strftime("%Y-%m-%d %H:%M:%S")}'
    )
    axes[2].set_title(
        f'GOES Merged (by max confidence)'
        f'\nProgression: {i+1} / {len(dates)}'
        f'\n'
    )
    plt.savefig(f'bobcat_imgs/{date.strftime("%Y-%m-%d_%H-%M-%S")}.png')
    plt.close()

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

def process_scene(file, target_area):
    '''
    - Reads the given nc file
    - Geolocates and resamples to the target area 
    - Converts mask values into confidence values
    '''
    high_quality_fdc_flags = [
        'good_quality_fire_pixel_qf' ,
        'good_quality_fire_free_land_pixel_qf' ,
        #'invalid_due_to_opaque_cloud_pixel_qf' ,
        #'invalid_due_to_surface_type_or_sunglint_or_LZA_threshold_exceeded_or_off_earth_or_missing_input_data_qf' ,
        #'invalid_due_to_bad_input_data_qf' ,
        #'invalid_due_to_algorithm_failure_qf' ,
    ]

    scn = Scene(
        reader='abi_l2_nc',
        filenames=[file],
        reader_kwargs={'filters': high_quality_fdc_flags}
    )
    scn.load(['Mask'])
    scn = scn.resample(target_area, resampler="nearest")
    scn = add_mask_confidence(scn)

    return scn

# NOTE extent should be saved to pkl file
# NOTE area_id should be saved to pkl file
def geolocate(west_files, east_files, dates, area_id, extent, plot_imgs):
    lon_min, lon_max, lat_min, lat_max = extent
    dlon, dlat = (0.01, 0.01) # native res ~ 0.02
    target_area = create_area_def(
        area_id=area_id,
        projection={"proj": "latlong", "datum": "WGS84"},
        area_extent=(lon_min, lat_min, lon_max, lat_max),
        resolution=(dlon, dlat),
        units="degrees"
    )

    fire_stack = []
    default_scn_west, default_scn_east = None, None

    for i in (pbar := tqdm(range(len(dates)))):
        w = west_files[i]
        e = east_files[i]
        d = dates[i]
        
        pbar.set_description(f'Processing {d}')
        if not w:
            scn_west = default_scn_west
        else:
            # west
            scn_west = process_scene(w, target_area)

        # east 
        if not e:
            scn_east = default_scn_east
        else:
            scn_east = process_scene(e, target_area)

        #### REVIEW
        # create a default scene of zeroes when there's downtime
        # requires the first time step to have a valid scene.
        # NOTE: implement patching (going back and filling zeros)?
        # maybe a warm-up period where we find the first valid nc file, grab its info to create the default scenes, then go into the main loop?
        if i == 0:
            default_scn_west = scn_west.copy()
            default_scn_west['MaskConfidence'] = xr.zeros_like(default_scn_west['MaskConfidence'])

            default_scn_east = scn_east.copy()
            default_scn_east['MaskConfidence'] = xr.zeros_like(default_scn_east['MaskConfidence'])
        ####
        
        # merged
        combined = xr.concat(
            [scn_west['MaskConfidence'], scn_east['MaskConfidence']],
            dim=xr.IndexVariable("satellite", ["GOES18", "GOES19"])
        )
        fire_stack.append(combined)

        # merge into one ds for the visualization
        reduced_min = combined.min(dim="satellite", skipna=True)
        #reduced_max = combined.max(dim="satellite", skipna=True)

        if plot_imgs:
            plot(scn_west, scn_east, reduced_min, d, i, 'bobcat_imgs')

    return fire_stack

with open('temp/filelist.pkl', 'rb') as f:
    goes_files = pickle.load(f)
    west_files = goes_files['west']
    east_files = goes_files['east']
    dates = goes_files['dates']
    outages = goes_files['outages']
    extent = goes_files['extent']
    area_id = goes_files['area_id']

buffer = 0.1 # 11.1km buffer, since exact extent risks a truncation
lon_min, lon_max, lat_min, lat_max = extent
extent = [
    lon_min - buffer,
    lon_max + buffer,
    lat_min - buffer,
    lat_max + buffer
]
fire_stack = geolocate(west_files, east_files, dates, area_id=area_id, extent=extent, plot_imgs=True)
combined_ds = combined_dataset(fire_stack, "temp/bobcat_max_combined.nc")
