from viz.gofer.ortho import make_original_vs_terrain_corrected_gif
from gofer.orthorectify import orthorectify
from argparse import ArgumentParser

argparser = ArgumentParser(description='Demonstration of the orthorectification on the Sierra Nevada.')
argparser.add_argument('--netcdf', help='The path of the netcdf file', default='/home/mgraca/Workspace/fire-spread/tests/gofer/data/OR_ABI-L1b-RadC-M3C02_G16_s20171110002189_e20171110004562_c20171110004596.nc')
argparser.add_argument('--tiff', help='The path of the tiff file', default='/home/mgraca/Workspace/fire-spread/data/dem/SRTMGL3_NC.003_SRTMGL3_DEM_doy2000042000000_aid0001.tif')
argparser.add_argument('--out', help='Path of the output gif', default='out/ortho/sierra_goes_original_vs_terrain_corrected.gif')
args = argparser.parse_args()

goes_file = args.netcdf
dem_file = args.tiff
out = args.out

ortho_ds = orthorectify(
    goes_filepath=goes_file,
    dem_filepath=dem_file,
    bbox=(-120.25, 37.25, -118.75, 38.50),
)

gif_path = make_original_vs_terrain_corrected_gif(
    goes_filepath=goes_file,
    ortho_ds=ortho_ds,
    variable="Rad",
    output_gif=out,
    cmap="gray",
    duration_ms=900,
)
