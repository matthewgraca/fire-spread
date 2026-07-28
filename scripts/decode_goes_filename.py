import pandas as pd
import argparse
from pathlib import Path

class Style:
    BOLD = '\033[1m'
    END = '\033[0m'

example_goes_file = 'OR_ABI-L2-FDCC-M3_G16_s20171440002078_e20171440004451_c20171440005297.nc'
parser = argparse.ArgumentParser(
    description=(
        'Takes a GOES file path, and grabs its components (UTC).'
        '\nExample:'
        f'\n\t $ python goes_time.py /mydata/{example_goes_file}'
        f'\n\t {example_goes_file} → 2017-05-24 00:05:29.700000'
    ),
    formatter_class=argparse.RawTextHelpFormatter
)

parser.add_argument(
    'file',
    help=f'GOES file path.\n\te.g. \'{example_goes_file}\'',
)
args = parser.parse_args()

file_name = str(Path(args.file).name)
system_env, product_mode, satellite, start, end, creation = file_name.rsplit('_', maxsplit=5)

start_ts = pd.to_datetime(start, format="s%Y%j%H%M%S%f")
end_ts = pd.to_datetime(end, format="e%Y%j%H%M%S%f")
creation_ts = pd.to_datetime(creation, format="c%Y%j%H%M%S%f.nc")

print(
    f'System Environment:   {Style.BOLD}{system_env}{Style.END}'
    f'\nProduct Mode:         {Style.BOLD}{product_mode}{Style.END}'
    f'\nStart scan:           {Style.BOLD}{start_ts} UTC{Style.END}'
    f'\nEnd scan:             {Style.BOLD}{end_ts} UTC{Style.END}'
    f'\nCreation time:        {Style.BOLD}{creation_ts} UTC{Style.END}'
)

