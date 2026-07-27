"""
Shared helpers for the GOFER pipeline scripts (ingest.py, run_pipeline.py).
"""
import geopandas as gpd
import pandas as pd
from tqdm import tqdm


# --- Configuration defaults ---

CALFIRE_GEOJSON = "data/calfire/California_Historic_Fire_Perimeters_-4891938132824355098.geojson"
GOES_SAVE_DIR = "data/goes"
DEM_FILEPATH = "data/dem/SRTMGL3_NC.003_SRTMGL3_DEM_doy2000042000000_aid0001.tif"
TEMP_BASE_DIR = "temp"
BBOX_BUFFER = 0.1


# --- Manifest ---

def load_manifest(manifest_path: str) -> pd.DataFrame:
    """
    Load and validate the fire manifest CSV.

    Expected columns: state, year, fire_name
    """
    df = pd.read_csv(manifest_path)
    required_cols = {'state', 'year', 'fire_name'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")
    df['fire_name'] = df['fire_name'].str.upper()
    return df


# --- Fire lookup ---

def lookup_fire(gdf: gpd.GeoDataFrame, fire_name: str, year: int):
    """
    Look up a fire from the CalFire GeoDataFrame by name and year.

    Returns the first matching row as a Series.
    """
    match = gdf.loc[
        (gdf['FIRE_NAME'] == fire_name) & (gdf['YEAR_'] == year)
    ]
    if len(match) == 0:
        raise ValueError(f"Fire '{fire_name}' ({year}) not found in CalFire data.")
    if len(match) > 1:
        tqdm.write(f"  Warning: multiple matches for '{fire_name}' ({year}), using first.")
    return match.iloc[0]
