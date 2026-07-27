from gofer.ingest import read_calfire_geojson
import geopandas as gpd

CALFIRE_GEOJSON = "data/calfire/California_Historic_Fire_Perimeters_-4891938132824355098.geojson"
min_year = 2020
min_acres = 50000
gdf = gpd.read_file(CALFIRE_GEOJSON)

gdf = gdf[
    (gdf['YEAR_'] >= min_year) &
    (gdf['GIS_ACRES'] > min_acres)
]
output = (
    gdf.rename(
        columns={
            "STATE": "state",
            "YEAR_": "year",
            "FIRE_NAME": "fire_name",
        }
    )
    .loc[:, ["state", "year", "fire_name"]]
    .assign(year=lambda x: x["year"].astype("Int64"))
)

output.to_csv('manifest/fires.csv', index=False)
