from gofer.ingest import download, read_calfire_geojson

calfire_geojson_path = "data/calfire/California_Historic_Fire_Perimeters_-4891938132824355098.geojson"
gdf = read_calfire_geojson(calfire_geojson_path)
bobcat_fire = gdf.loc[gdf['FIRE_NAME'] == 'BOBCAT']

bobcat_fire_attrs = {
    'start' :bobcat_fire['ALARM_DATE'].item(),
    'end' : bobcat_fire['CONT_DATE'].item(),
    'fire_name' : str(bobcat_fire['FIRE_NAME'].item()),
    'fire_year' : int(bobcat_fire['YEAR_'].item()),
    'fire_acres' : int(bobcat_fire['GIS_ACRES'].item())
}
bobcat_fire_bbox = {
    'lon_min' : float(bobcat_fire['bbox_min_lon'].item()),
    'lon_max' : float(bobcat_fire['bbox_max_lon'].item()),
    'lat_min' : float(bobcat_fire['bbox_min_lat'].item()),
    'lat_max' : float(bobcat_fire['bbox_max_lat'].item())
}

download(
    **bobcat_fire_attrs,
    goes_save_dir='/home/mgraca/Workspace/fire-spread/data/goes',
    metadata_save_dir='temp',
    subhourly=True,
    **bobcat_fire_bbox
)
