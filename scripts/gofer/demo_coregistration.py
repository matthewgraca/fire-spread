from gofer.orthorectify import orthorectify
from argparse import ArgumentParser
import pkl

argparser = ArgumentParser(description='Demonstration of the orthorectification on the Sierra Nevada.')
argparser.add_argument('--west', help='The path of the GOES-West netcdf file', default='/home/mgraca/Workspace/fire-spread/gofer/data/noaa-goes17/ABI-L2-FDCC/2020/270/20/OR_ABI-L2-FDCC-M6_G17_s20202702001176_e20202702003549_c20202702004130.nc')
argparser.add_argument('--east', help='The path of the GOES-East netcdf file', default='/home/mgraca/Workspace/fire-spread/gofer/data/noaa-goes16/ABI-L2-FDCC/2020/270/20/OR_ABI-L2-FDCC-M6_G16_s20202702001167_e20202702003540_c20202702004310.nc')
argparser.add_argument('--tiff', help='The path of the DEM tiff file', default='/home/mgraca/Workspace/fire-spread/data/dem/SRTMGL3_NC.003_SRTMGL3_DEM_doy2000042000000_aid0001.tif')
argparser.add_argument('--pkl', help='Pickle file containing fire metadata', default='/home/mgraca/Workspace/fire-spread/temp/filelist.pkl')
args = argparser.parse_args()

with open(args.pkl, 'rb') as f:
    goes_files = pickle.load(f)
    west_files = goes_files['west']
    east_files = goes_files['east']
    dates = goes_files['dates']
    outages = goes_files['outages']
    extent = goes_files['extent']
    area_id = goes_files['area_id']

buffer = 0.1
lon_min, lon_max, lat_min, lat_max = extent
bbox = (
    lon_min - buffer,
    lat_min - buffer,
    lon_max + buffer,
    lat_max + buffer
)

west_ortho_ds = orthorectify(
    goes_filepath=args.west,
    dem_filepath=args.tiff,
    bbox=bbox,
)

print(west_ortho_ds)

'''
east_ortho_ds = orthorectify(
    goes_filepath=args.east,
    dem_filepath=args.tiff,
    bbox=bbox,
)
'''

# NOTE reopening the tiff is silly, lets make this a one-time affair, but leave it at the end. 
# NOTE need to convert the data into mask confidence here too. merge then compute = vectorized?
# NOTE merge all into one dataset; see combine_dataset() in process_nc_files.py
# left, right, mixed viz of final perimeter, maybe can just use the fire_perimeter.py script here?.
