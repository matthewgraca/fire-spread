import pandas as pd
import geopandas as gpd
import pickle
from pathlib import Path
from goes2go.data import goes_nearesttime, goes_timerange
from tqdm import tqdm
import contextlib
import io
import traceback

#### multi-row operations; returns new dataframe
def _add_fire_centroids(gdf: gpd.GeoDataFrame):
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

def _add_fire_bboxes(gdf: gpd.GeoDataFrame):
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


def read_calfire_geojson(filepath: str, min_year: int = 2020, min_acres: int = 20000) -> gpd.GeoDataFrame:
    '''
    Reads and grabs relevant data from the CalFire geojson.


    GeoJSON data found at:
    https://data.ca.gov/dataset/california-fire-perimeters-1950
    '''
    gdf = gpd.read_file(filepath)

    big_fires = gdf[
        (gdf['YEAR_'] >= min_year) &
        (gdf['GIS_ACRES'] > min_acres)
    ]

    temp_gdf = _add_fire_bboxes(big_fires)
    temp_gdf = _add_fire_centroids(temp_gdf)

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

    return temp_gdf


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
        'verbose' : verbose,
        'ignore_missing' : True
    }
    def _goes2go_ingest(subhourly, date, goes2go_kwargs):
        if subhourly:
            g = goes_timerange(
                start=date,
                end=date + pd.Timedelta(hours=1),
                **goes2go_kwargs
            )
        else:
            g = goes_nearesttime(attime=date, **goes2go_kwargs)
        return g

    error_hit = False
    if silent:
        buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(buffer):
                g = _goes2go_ingest(subhourly, date, goes2go_kwargs)
        # Missing on AWS's end
        except FileNotFoundError as e:
            tqdm.write(f'Data for {date} missing for GOES-{satellite}.')
            error_hit = True
        except KeyboardInterrupt:
            raise
        except:
            tqdm.write(f'Data for {date} missing/corrupted for GOES-{satellite}.')
            error_hit = True
    else:
        g = _goes2go_ingest(subhourly, date, goes2go_kwargs)

    return None if error_hit else g


def download(
    start, end,
    goes_save_dir, metadata_save_dir, subhourly,
    lon_min, lon_max, lat_min, lat_max,
    fire_name, fire_year, fire_acres
):
    tqdm.write(f'Ingesting for {fire_name} {fire_year}')
    dates = pd.date_range(start, end, freq='h', inclusive='left')
    ingest_dates = dates.shift(-1) if subhourly else dates

    goes_kwargs = {
        'product' : 'ABI-L2-FDCC',
        'domain' : 'C',
        'save_dir' : goes_save_dir,
        #'return_as' : 'filelist',
        'verbose' : False,
        'subhourly' : subhourly,
        'silent' : True
    }
    west_files = []
    east_files = []

    for d in (pbar := tqdm(ingest_dates.tz_localize(None))):
        pbar.set_description(f'Ingesting GOES-East and GOES-West on {d}')

        east_file_df = ingest(d, satellite='EAST', **goes_kwargs)
        if east_file_df is not None:
            east_files.append(east_file_df)

        west_file_df = ingest(d, satellite='WEST', **goes_kwargs)
        if west_file_df is not None:
            west_files.append(west_file_df)

    Path(metadata_save_dir).mkdir(parents=True, exist_ok=True)
    tqdm.write(f'Saving file information, dates, fire metadata, and bounding box to {str(metadata_save_dir)}')
    with open(metadata_save_dir / Path('metadata.pkl'), 'wb') as f:
        pkg = {
            'dates' : dates,
            'fire_attrs' : {
                'name' : fire_name,
                'year' : fire_year,
                'acres' : fire_acres
            },
            'lon_min' : lon_min,
            'lon_max' : lon_max,
            'lat_min' : lat_min,
            'lat_max' : lat_max
        }
        pickle.dump(pkg, f)

    west_files_df = pd.concat(west_files, ignore_index=True)
    east_files_df = pd.concat(east_files, ignore_index=True)

    west_files_df.to_csv(metadata_save_dir / Path('west_files.csv'), index=False) 
    east_files_df.to_csv(metadata_save_dir / Path('east_files.csv'), index=False) 

    return None
