import xarray as xr
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pickle
from tqdm import tqdm
from viz.gofer.tilers import CartoDBTiles

with open('temp/bobcat_2020/metadata.pkl', 'rb') as f:
    goes_files = pickle.load(f)
    dates = goes_files['dates']
    buffer = 0.1
    bbox = (
        goes_files['lon_min'] - buffer,
        goes_files['lat_min'] - buffer,
        goes_files['lon_max'] + buffer,
        goes_files['lat_max'] + buffer
    )
    extent = [
        goes_files['lon_min'] - buffer,
        goes_files['lon_max'] + buffer,
        goes_files['lat_min'] - buffer,
        goes_files['lat_max'] + buffer
    ]

combined_ds = xr.open_dataset('out/bobcat_2020_gofer.nc', chunks='auto')
print(combined_ds)

plot_shared_kwargs = {
    'transform' : ccrs.PlateCarree(),
    'cmap' : 'YlOrRd',
    'vmin' : 0.0,
    'vmax' : 1.0,
    'add_colorbar' : False
}

#print(combined_ds["time"])
'''
time_start = combined_ds["time"].coarsen(time=12, boundary="trim").min()
time_end = combined_ds["time"].coarsen(time=12, boundary="trim").max()
# groups into non-overlapping 12 frame chunks
# same, but a cumulative max
ds = (
    combined_ds.coarsen(time=12, boundary='trim')
    .max()
    .cumulative(dim='time')
    .max()
)
ds = ds.assign_coords(
    time_start=("time", time_start.data),
    time_end=("time", time_end.data),
)
'''
#print(ds)

gdf = gpd.read_file("data/calfire/California_Historic_Fire_Perimeters_-4891938132824355098.geojson")
bobcat_fire = gdf.loc[gdf['FIRE_NAME'] == 'BOBCAT']

''' full
for i, d in tqdm(enumerate(ds['time'].values)):
    tiler = CartoDBTiles(style='rastertiles/voyager', cache=True)
    fig, axes = plt.subplots(1, 1, figsize=(16, 12), subplot_kw={'projection' : ccrs.PlateCarree()}, layout='constrained')

    mask_conf = ds["MaskConfidence"].isel(time=i)
    plot = mask_conf.where(mask_conf != 0).plot(ax=axes, **plot_shared_kwargs)

    axes.add_image(tiler, 12)
    axes.set_extent(extent, crs=ccrs.PlateCarree())
    bobcat_fire.to_crs(epsg=4326).plot(
        ax=axes,
        transform=ccrs.PlateCarree(),
        facecolor='none',
        edgecolor='black'
    )

    cbar = fig.colorbar(
        plot,
        ax=axes,
        orientation='vertical',
        shrink=0.7,
        pad=0.03
    )

    cbar.set_label("Fire Confidence")

    dt_start = pd.to_datetime(mask_conf['time_start'].item())
    dt_end = pd.to_datetime(mask_conf['time_end'].item())

    axes.set_title(
        f"GOFER \n"
        f"start: {dt_start}\n"
        f"end: {dt_end}"
    )

    plt.savefig(f"out/gofer/{dt_end.strftime('%Y-%m-%dT%H:%M:%S')}.png")
    plt.close()
'''
''' just final '''
tiler = CartoDBTiles(style='rastertiles/voyager', cache=True)
fig, axes = plt.subplots(1, 1, figsize=(16, 12), subplot_kw={'projection' : ccrs.PlateCarree()}, layout='constrained')

# cummax over whole ds
#ds_cummax = combined_ds['MaskConfidence'].cumulative('time').max()
mask_conf = combined_ds['MaskConfidence'].isel(time=-1)
# plot high-confidence only
plot = mask_conf.where(mask_conf).plot(ax=axes, **plot_shared_kwargs)

axes.add_image(tiler, 12)
axes.set_extent(extent, crs=ccrs.PlateCarree())
bobcat_fire.to_crs(epsg=4326).plot(
    ax=axes,
    transform=ccrs.PlateCarree(),
    facecolor='none',
    edgecolor='black'
)

cbar = fig.colorbar(
    plot,
    ax=axes,
    orientation='vertical',
    shrink=0.7,
    pad=0.03
)

cbar.set_label("Fire Confidence")

'''
dt_start = pd.to_datetime(mask_conf['time'].item())
dt_end = pd.to_datetime(mask_conf['time_end'].item())

axes.set_title(
    f"GOFER \n"
    f"start: {dt_start}\n"
    f"end: {dt_end}"
)

plt.savefig(f"out/gofer/{dt_end.strftime('%Y-%m-%dT%H:%M:%S')}.png")
'''
axes.set_title(f"GOFER")
plt.savefig(f"out/gofer/bobcat.png")
plt.close()
