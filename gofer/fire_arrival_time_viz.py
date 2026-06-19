import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pickle
import numpy as np
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

# merge into one array
# NOTE need to verify this...
# purple: old fire, red: new fire
# i think this is being messed up by the confidence scores? it assumes binary mask
# FIRE ARRIVAL TIME
def encode_oldest_on_top_dataset(ds, var="MaskConfidence", dim="time", sentinel=0):
    da = ds[var]
    n = da.sizes[dim]

    # Highest priority goes to oldest layer.
    # time=0 -> n
    # time=n-1 -> 1
    priority = xr.DataArray(
        np.arange(n, 0, -1),
        dims=dim,
        coords={dim: da[dim]},
    )

    # Actual output values you want:
    # time=0 -> 1/n
    # time=n-1 -> 1
    time_values = xr.DataArray(
        np.arange(1, n + 1) / n,
        dims=dim,
        coords={dim: da[dim]},
    )

    # For every nonzero cell, store its priority.
    # For zero cells, store 0.
    priority_encoded = xr.where(da != sentinel, priority, 0)

    # Oldest nonzero cell wins because it has the largest priority.
    winning_priority = priority_encoded.max(dim=dim)

    # Convert winning priority back to normalized time value.
    out = (n - winning_priority + 1) / n

    # Cells that were never present should remain 0.
    out = out.where(winning_priority != 0, 0)

    return out

out_min = encode_oldest_on_top_dataset(combined_ds.min(dim="satellite", skipna=True))
out_max = encode_oldest_on_top_dataset(combined_ds.max(dim="satellite", skipna=True))

# plot
gdf = gpd.read_file("calfire/California_Historic_Fire_Perimeters_-4891938132824355098.geojson")
bobcat_fire = gdf.loc[gdf['FIRE_NAME'] == 'BOBCAT']

# also consider naming it "conservative" vs "liberal" observations
tiler = CartoDBTiles(style='rastertiles/voyager', cache=True)
fig, axes = plt.subplots(1, 2, figsize=(12, 8), subplot_kw={'projection' : ccrs.PlateCarree()}, layout='constrained')

min_plot = out_min.where(out_min != 0).plot(ax=axes[0], **plot_shared_kwargs)
max_plot = out_max.where(out_max != 0).plot(ax=axes[1], **plot_shared_kwargs)

for ax in axes:
    ax.add_image(tiler, 12)
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    bobcat_fire.to_crs(epsg=4326).plot(ax=ax, transform=ccrs.PlateCarree(), facecolor='none', edgecolor='black')

cbar = fig.colorbar(
    max_plot,
    ax=axes,
    orientation='vertical',
    shrink=0.7,
    pad=0.03
)

cbar.set_label("Fire Arrival Time")

axes[0].set_title(
    f'GOES Merged (by min confidence; conservative)'
    f'\nstart: {dates[0]}'
    f'\nend: {dates[-1]}'
)
axes[1].set_title(
    f'GOES Merged (by max confidence; liberal)'
    f'\nstart: {dates[0]}'
    f'\nend: {dates[-1]}'
)
plt.savefig('out_imgs/bobcat_arrival_time_map.png')
plt.show()


# INDIVIDUAL PLOTS


west = encode_oldest_on_top_dataset(combined_ds.sel(satellite='GOES18'))
east = encode_oldest_on_top_dataset(combined_ds.sel(satellite='GOES19'))

# plot
gdf = gpd.read_file("calfire/California_Historic_Fire_Perimeters_-4891938132824355098.geojson")
bobcat_fire = gdf.loc[gdf['FIRE_NAME'] == 'BOBCAT']

# also consider naming it "conservative" vs "liberal" observations
# NOTE: over prediction due to low-confidence fire pixels being accepted
fig, axes = plt.subplots(1, 2, figsize=(12, 8), subplot_kw={'projection' : ccrs.PlateCarree()}, layout='constrained')

west_plot = west.where(west != 0).plot(ax=axes[0], **plot_shared_kwargs)
east_plot = west.where(east != 0).plot(ax=axes[1], **plot_shared_kwargs)

for ax in axes:
    ax.add_image(tiler, 12)
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    bobcat_fire.to_crs(epsg=4326).plot(ax=ax, transform=ccrs.PlateCarree(), facecolor='none', edgecolor='black')

cbar = fig.colorbar(
    west_plot,
    ax=axes,
    orientation='vertical',
    shrink=0.7,
    pad=0.03
)

cbar.set_label("Fire Arrival Time")

axes[0].set_title(
    f'GOES-West'
    f'\nstart: {dates[0]}'
    f'\nend: {dates[-1]}'
)
axes[1].set_title(
    f'GOES-East'
    f'\nstart: {dates[0]}'
    f'\nend: {dates[-1]}'
)
plt.savefig('out_imgs/bobcat_arrival_time_map_isolated.png')
plt.show()
