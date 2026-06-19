import pandas as pd
import geopandas as gpd
import pickle

from goes2go.data import goes_nearesttime
from tqdm import tqdm
import contextlib
import io
import traceback

# NOTE to make this into a proper class, we need to know the input. 
# will it ingest all fires at once? do we pass in geodataframe of the exact fires we want?

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
calfire_geojson_path = "calfire/California_Historic_Fire_Perimeters_-4891938132824355098.geojson"
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
trimmed_cols = [
    'YEAR_', 'FIRE_NAME', 'ALARM_DATE',
    'CONT_DATE', 'GIS_ACRES', 'COMPLEX_NAME', 'geometry', 'bbox_min_lon',
    'bbox_min_lat', 'bbox_max_lon', 'bbox_max_lat', 'centroid_lon',
    'centroid_lat'
]

# NOTE this is where you select the fire you want. we do it by name, but we can change the method by which we grab fires
# like a whole dataframe, then iterate through them for the ingestion portion
bobcat_fire = temp_gdf.loc[temp_gdf['FIRE_NAME'] == 'BOBCAT']
print(bobcat_fire)

# INGESTION
def ingest(date, satellite, product, domain, save_dir, verbose, silent):
    nearest_time_kwargs = {
        'attime' : date,
        'satellite' : satellite,
        'product' : product,
        'domain' : domain,
        'save_dir' : save_dir,
        'return_as' : 'filelist',
        'verbose' : verbose
    }
    file = ''
    if silent:
        buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(buffer):
                g = goes_nearesttime(**nearest_time_kwargs)
                file = g['file'].item()
        # Missing on AWS's end
        except FileNotFoundError as e:
            tqdm.write(f'Data for {date} missing for GOES-{satellite}.')
        except:
            raise
    else:
        g = goes_nearesttime(**nearest_time_kwargs)
        file = g['file'].item()

    return file

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
        if not e:
            outages['EAST'].append(d)
        if not w:
            outages['WEST'].append(d)

    return outages

tqdm.write(f'Ingesting for {bobcat_fire["FIRE_NAME"].item()}')
dates = pd.date_range(
    bobcat_fire['ALARM_DATE'].item(),
    bobcat_fire['CONT_DATE'].item(),
    freq='h',
    inclusive='left'
).tz_localize(None)
goes_save_dir = '/home/mgraca/Workspace/goes-projection'
goes_kwargs = {
    'product' : 'ABI-L2-FDCC',
    'domain' : 'C',
    'save_dir' : goes_save_dir,
    #'return_as' : 'filelist',
    'verbose': False
}
west_files = []
east_files = []
for d in (pbar := tqdm(dates)):
    pbar.set_description(f'Ingesting GOES-East and GOES-West on {d}')
    east_files.append(ingest(d, 'EAST', **goes_kwargs, silent=True))
    west_files.append(ingest(d, 'WEST', **goes_kwargs, silent=True))

# need logic to handle missing frames (either one or both)
outages = get_outages(east_files, west_files, dates)

pkl_filepath = 'filelist.pkl'
tqdm.write(f'Saving west/east filepaths, dates, and outages to {pkl_filepath}')
with open(pkl_filepath, 'wb') as f:
    pkg = {
        'west' : west_files,
        'east' : east_files,
        'dates' : dates,
        'outages' : outages
    }
    pickle.dump(pkg, f)
