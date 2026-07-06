import pandas as pd
import geopandas as gpd
import pickle
from pathlib import Path
from goes2go.data import goes_nearesttime, goes_timerange
from tqdm import tqdm
import contextlib
import io
import traceback

# NOTE to make this into a proper class, we need to know the input. 
# will it ingest all fires at once? do we pass in geodataframe of the exact fires we want?
# I think that no matter what, the ingest() function will need to should take in:
#   start and end date, name, year, acres, bbox

#### multi-row operations; returns new dataframe
def add_fire_centroids(gdf: gpd.GeoDataFrame):
    '''
    Finds the centroids of fires in the dataset, and adds them to the dataset.
    '''
    # Project to California Albers for better centroid calculations
    temp_gdf = gdf.copy()
    temp_gdf = temp_gdf.to_crs("EPSG:3310")
    centroids_projected = temp_gdf.geometry.centroid

    # Convert centroid points back to lon/lat
    centroids_lonlat = gpd.GeoSeries(
        centroids_projected,
        crs="EPSG:3310",
        index=gdf.index
    ).to_crs("EPSG:4326")

    # Add columns to original GeoDataFrame
    temp_gdf["centroid_lon"] = centroids_lonlat.x
    temp_gdf["centroid_lat"] = centroids_lonlat.y

    return temp_gdf

def add_fire_bboxes(gdf: gpd.GeoDataFrame):
    '''
    Finds a fire's bounding box, and adds them to the dataset

    Arguments:
        gdf (GeoDataFrame): A GeoDataFrame that contains the fire information.

    Returns:
        GeoDataFrame: GeoDataframe with the added columns: 
        min_lon, min_lat, max_lon, max_lat
    '''
    temp_gdf = gdf.copy()
    temp_gdf = temp_gdf.to_crs(4326)

    bounds = temp_gdf.geometry.bounds
    bounds = bounds.rename(columns={
        "minx": "bbox_min_lon",
        "miny": "bbox_min_lat",
        "maxx": "bbox_max_lon",
        "maxy": "bbox_max_lat",
    })
    temp_gdf = temp_gdf.join(bounds)

    return temp_gdf

####

'''
GeoJSON data found at:
https://data.ca.gov/dataset/california-fire-perimeters-1950
'''
# NOTE param
calfire_geojson_path = "data/calfire/California_Historic_Fire_Perimeters_-4891938132824355098.geojson"
gdf = gpd.read_file(calfire_geojson_path)

big_fires = gdf[
    (gdf['YEAR_'] >= 2020) &
    (gdf['GIS_ACRES'] > 20000)
]

temp_gdf = add_fire_bboxes(big_fires)
temp_gdf = add_fire_centroids(temp_gdf)
print(temp_gdf)
print(temp_gdf.columns)

'''
['OBJECTID', 'YEAR_', 'STATE', 'AGENCY', 'UNIT_ID', 'FIRE_NAME',
       'INC_NUM', 'IRWINID', 'ALARM_DATE', 'CONT_DATE', 'CAUSE', 'C_METHOD',
       'OBJECTIVE', 'GIS_ACRES', 'COMMENTS', 'COMPLEX_NAME', 'COMPLEX_ID',
       'FIRE_NUM', 'GlobalID', 'DECADES', 'geometry', 'bbox_min_lon',
       'bbox_min_lat', 'bbox_max_lon', 'bbox_max_lat', 'centroid_lon',
       'centroid_lat']
'''
metadata_cols = ['FIRE_NAME', 'YEAR_', 'GIS_ACRES']
trimmed_cols = [
    'FIRE_NAME', 'ALARM_DATE',
    'CONT_DATE', 'GIS_ACRES', 'COMPLEX_NAME', 'geometry', 'bbox_min_lon',
    'bbox_min_lat', 'bbox_max_lon', 'bbox_max_lat', 'centroid_lon',
    'centroid_lat'
]

# NOTE this is where you select the fire you want. we do it by name, but we can change the method by which we grab fires
# like a whole dataframe, then iterate through them for the ingestion portion
bobcat_fire = temp_gdf.loc[temp_gdf['FIRE_NAME'] == 'BOBCAT']

# INGESTION
def ingest(date, subhourly, satellite, product, domain, save_dir, verbose, silent):
    '''
    Ingests the GOES-West and GOES-East data from NOAA's AWS bucket.

    Args:
        date (datetime64[ns]): The date.
        subhourly (bool): If true, all observations will be ingested. If 
            false, only the nearest observation to the hour will be taken.
        satellite (str): The satellite to grab the data from.
        product (str): The specific GOES product to be grabbed.
        domain (str): The domain (mesoscale, conus, full disk).
        save_dir (str): The directory the data will be saved to.
        verbose (bool): Verbosity.
        silent (bool): Whether to silence the goes2go messages (they can get 
            annoyingly loud).

    Returns:
        pd.DataFrame: A dataframe of the file download. Expect:
        file, product_mode, satellite, start, end, creation, product, 
        mode_bands, mode, band

    '''
    goes2go_kwargs = {
        'satellite' : satellite,
        'product' : product,
        'domain' : domain,
        'save_dir' : save_dir,
        'return_as' : 'filelist',
        'verbose' : verbose
    }
    # NOTE replace nearest time with all. replace files with empty strings? Will need to change logic in...
    #grep -R --include="*.py" "pickle" that looks for empty files (composite)? 
    # NOTE redo composite tests and pipeline demo!!! could move those to test out remapper instead of binning it.
    def _goes2go_ingest(subhourly, goes2go_kwargs):
        if subhourly:
            g = goes_timerange(
                start=date,
                end=date + pd.Timedelta(hours=1),
                **goes2go_kwargs
            )
        else:
            g = goes_nearesttime(attime=date, **goes2go_kwargs)
        return g

    if silent:
        buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(buffer):
                g = _goes2go_ingest(subhourly, goes2go_kwargs)
        # Missing on AWS's end
        except FileNotFoundError as e:
            tqdm.write(f'Data for {date} missing for GOES-{satellite}.')
        except:
            raise
    else:
        g = _goes2go_ingest(subhourly, goes2go_kwargs)

    return g

def get_outages(east, west, dates):
    if len(east) != len(west) != len(dates):
        raise ValueError(
            f'Number of files mismatch (East = {len(east)}, '
            'West = {len(west)}, Dates = {len(dates)}'
        )
    outages = {
        'EAST' : [],
        'WEST' : []
    }
    for e, w, d in zip(east, west, dates):
        if not Path(e).is_file():
            outages['EAST'].append(d)
        if not Path(w).is_file():
            outages['WEST'].append(d)

    return outages

tqdm.write(f'Ingesting for {bobcat_fire["FIRE_NAME"].item()}')
dates = pd.date_range(
    bobcat_fire['ALARM_DATE'].item(),
    bobcat_fire['CONT_DATE'].item(),
    freq='h',
    inclusive='left'
)
print(dates)
# NOTE param
goes_save_dir = '/home/mgraca/Workspace/fire-spread/data/goes'
goes_kwargs = {
    'product' : 'ABI-L2-FDCC',
    'domain' : 'C',
    'save_dir' : goes_save_dir,
    #'return_as' : 'filelist',
    'verbose': False
}
west_files = []
east_files = []

# NOTE TEST
dates = dates[:30]

# use one hour before for subhourly
subhourly = True
ingest_dates = dates.shift(-1) if subhourly else dates
print(ingest_dates)
for d in (pbar := tqdm(ingest_dates.tz_localize(None))):
    pbar.set_description(f'Ingesting GOES-East and GOES-West on {d}')
    east_files.append(ingest(d, subhourly=subhourly, satellite='EAST', **goes_kwargs, silent=True))
    west_files.append(ingest(d, subhourly=subhourly, satellite='WEST', **goes_kwargs, silent=True))

# NOTE param
pkl_filepath = 'temp/metadata.pkl'
Path(pkl_filepath).parent.mkdir(parents=True, exist_ok=True)
tqdm.write(f'Saving dates, fire metadata, and bounding box to {pkl_filepath}')
with open(pkl_filepath, 'wb') as f:
    pkg = {
        'dates' : dates,
        'fire_attrs' : {
            'name' : str(bobcat_fire['FIRE_NAME'].item()),
            'year' : int(bobcat_fire['YEAR_'].item()),
            'acres' : int(bobcat_fire['GIS_ACRES'].item())
        },
        'lon_min' : float(bobcat_fire['bbox_min_lon'].item()),
        'lon_max' : float(bobcat_fire['bbox_max_lon'].item()),
        'lat_min' : float(bobcat_fire['bbox_min_lat'].item()),
        'lat_max' : float(bobcat_fire['bbox_max_lat'].item())
    }
    pickle.dump(pkg, f)

west_files_df = pd.concat(west_files, ignore_index=True)
print(west_files_df)
east_files_df = pd.concat(east_files, ignore_index=True)
print(east_files_df)
west_files_df.to_csv('temp/west_files.csv', index=False) 
east_files_df.to_csv('temp/east_files.csv', index=False) 
