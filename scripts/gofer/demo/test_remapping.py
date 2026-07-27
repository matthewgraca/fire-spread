from gofer.ortho import orthorectify
from gofer.spatial_smoothing import smooth
import xarray as xr
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

with open("temp/metadata.pkl", "rb") as f:
    data = pickle.load(f)

buffer = 0.1
bbox = (
    data["lon_min"] - buffer,
    data["lat_min"] - buffer,
    data["lon_max"] + buffer,
    data["lat_max"] + buffer,
)

dem_filepath = (
    "/home/mgraca/Workspace/fire-spread/data/dem/"
    "SRTMGL3_NC.003_SRTMGL3_DEM_doy2000042000000_aid0001.tif"
)
print(data['dates'])
import sys
sys.exit()

with xr.open_dataset("temp/west_bobcat_2020_aggregated.nc", decode_times=False, chunks={'time': 1}) as src:
    print(src)

    # Force data to be read while the source file is definitely open.
    out = orthorectify(
        src,
        dem_filepath=dem_filepath,
        bbox=bbox,
        data_var="MaskConfidence",
    )

print(out)

for i in tqdm(range(0, 25)):
    out['MaskConfidence'].isel(time=i).plot()
    plt.savefig(f'temp/imgs/early_perimeter/west/{i:02d}_ortho.png')
    plt.close()

out = smooth(out)
print(out)

for i in tqdm(range(0, 25)):
    out['MaskConfidence'].isel(time=i).plot()
    plt.savefig(f'temp/imgs/early_perimeter/west/{i:02d}_ortho_and_smoothed.png')
    plt.close()
