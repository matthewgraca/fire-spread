import xarray as xr
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pickle
from tqdm import tqdm
from .tilers import CartoDBTiles

with open('temp/filelist.pkl', 'rb') as f:
    goes_files = pickle.load(f)
    west_files = goes_files['west']
    east_files = goes_files['east']
    dates = goes_files['dates']
    outages = goes_files['outages']
    extent = goes_files['extent']
    area_id = goes_files['area_id']

combined_ds = xr.open_dataset('temp/bobcat_max_combined.nc')
buffer = 0.1 # 11.1km buffer, since exact buffer might truncate
lon_min, lon_max, lat_min, lat_max = extent
extent = [
    lon_min - buffer,
    lon_max + buffer,
    lat_min - buffer,
    lat_max + buffer
]

plot_shared_kwargs = {
    'transform' : ccrs.PlateCarree(),
    'cmap' : 'YlOrRd',
    'vmin' : 0.0,
    'vmax' : 1.0,
    'add_colorbar' : False
}

time_start = combined_ds["time"].coarsen(time=12, boundary="trim").min()
time_end = combined_ds["time"].coarsen(time=12, boundary="trim").max()
# groups into non-overlapping 12 frame chunks
'''
ds = (
    combined_ds.coarsen(time=12, boundary='trim')
    .max()
)
'''
# same, but a cumulative max
ds = (
    combined_ds.coarsen(time=12, boundary='trim')
    .max()
    .cumulative(dim='time')
    .max()
)
# this MERGES all 12 frame chunks, NOT skipping! that's:
#ds = ds.isel(time=slice(None, None, 12))
ds = ds.assign_coords(
    time_start=("time", time_start.data),
    time_end=("time", time_end.data),
)
print(ds)

gdf = gpd.read_file("calfire/California_Historic_Fire_Perimeters_-4891938132824355098.geojson")
bobcat_fire = gdf.loc[gdf['FIRE_NAME'] == 'BOBCAT']

for i, d in tqdm(enumerate(ds['time'].values)):
    tiler = CartoDBTiles(style='rastertiles/voyager', cache=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), subplot_kw={'projection' : ccrs.PlateCarree()}, layout='constrained')

    # merge into one ds
    west = ds['MaskConfidence'].sel(satellite='GOES18').isel(time=i)
    west_data = west.where(west != 0)

    east = ds['MaskConfidence'].sel(satellite='GOES19').isel(time=i)
    east_data = east.where(east != 0)

    reduced_max = ds['MaskConfidence'].max(dim='satellite', skipna=True).isel(time=i)
    max_data = reduced_max.where(reduced_max != 0)

    reduced_min = ds['MaskConfidence'].min(dim='satellite', skipna=True).isel(time=i)
    min_data = reduced_min.where(reduced_min != 0)

    west_plot = west_data.plot(ax=axes[0][0], **plot_shared_kwargs)
    east_plot = east_data.plot(ax=axes[0][1], **plot_shared_kwargs)
    max_plot = max_data.plot(ax=axes[1][0], **plot_shared_kwargs)
    min_plot = min_data.plot(ax=axes[1][1], **plot_shared_kwargs)

    for ax in axes.flatten():
        ax.add_image(tiler, 12)
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        bobcat_fire.to_crs(epsg=4326).plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            facecolor='none',
            edgecolor='black'
        )

    cbar = fig.colorbar(
        west_plot,
        ax=axes,
        orientation='vertical',
        shrink=0.7,
        pad=0.03
    )

    cbar.set_label("Fire Confidence")

    axes[0][0].set_title(
        f'GOES-West'
        f'\nstart: {pd.to_datetime(west_data.time_start.item()).strftime("%Y-%m-%d %H:%M:%S")}'
        f'\nend: {pd.to_datetime(west_data.time_end.item()).strftime("%Y-%m-%d %H:%M:%S")}'
    )
    axes[0][1].set_title(
        f'GOES-East'
        f'\nstart: {pd.to_datetime(east_data.time_start.item()).strftime("%Y-%m-%d %H:%M:%S")}'
        f'\nend: {pd.to_datetime(east_data.time_end.item()).strftime("%Y-%m-%d %H:%M:%S")}'
    )
    axes[1][0].set_title(
        f'GOES Merged (by union; liberal)'
        f'\nProgression: {i+1} / {len(ds.time.values)}'
        f'\n'
    )
    axes[1][1].set_title(
        f'GOES Merged (by intersection; conservative)'
        f'\nProgression: {i+1} / {len(ds.time.values)}'
        f'\n'
    )
    plt.savefig(f'bobcat_imgs_cummax_12/{pd.to_datetime(east_data.time_end.item()).strftime("%Y-%m-%d_%H-%M-%S")}.png')
    plt.close()
