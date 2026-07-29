# Configs
## GOFER
Description: Responsible for all the options necessary to run the GOFER pipeline.

Options:
- `manifest`: The path of the csv that lists the fires to ingest and process.
    - Example: `/home/mgraca/Workspace/fire-spread/manifests/fires.csv`
- `calfire_geojson`: The path of the CalFire GeoJSON that will be used to grab fire metadata and final perimeters.
    - Example: `/mnt/wildfire/fire-spread/calfire/California_Historic_Fire_Perimeters_-4891938132824355098.geojson`
- `goes_dir`: The directory where the ingested raw GOES data will live.
    - Example: `/mnt/wildfire/fire-spread/goes`
- `temp_dir`: The directory where the intermediate netcdf files from the GOFER pipeline will live.
    - Example: `/mnt/wildfire/fire-spread/temp`
- `out_dir`: The directory where the final product will live.
    - Example: `/mnt/wildfire/fire-spread/out`
- `dem`: The path of the digital elevation map that will be used for orthorectification.
    - Example: `/mnt/wildfire/fire-spread/dem/SRTMGL3_NC.003_SRTMGL3_DEM_doy2000042000000_aid0001.tif`
- `clean`: Whether or not to delete the contents of the temp folder after processing. 
    - Example: `true`
    - Recommendation: Keep `true`, the intermediate files can get large.
- `memory_limit`: String detailing the amount of memory you want allocated in RAM to run the GOFER pipeline. Dask will handle the rest.
    - Example: `50GB`
    - Recommendation: Your available RAM - 14GB
- `threads`: Integer detailing the number of threads you want Dask to allocate to the job.
    - Example: `12`
    - Recommendation: Your available threadcount // 2
    - Note: Since this script is expected to have limited users, we didn't feel the need to automagically determine these parameters.
